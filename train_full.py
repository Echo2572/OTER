import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from encoder import FullOTER, FullRelationModel
from encoder.fewrel_loader import FewRELFewShotLoader
from encoder.mnre_loader import DynamicFewShotLoader


LOGGER = logging.getLogger("encoder")
PACKAGE_ROOT = Path(__file__).resolve().parent


PROFILES = {
    "mnre": {
        "data_root": "/home/t5820/syb/data",
        "seed": 77,
        "eval_seed": 3407,
        "test_seed": 3407,
        "split_path": "/home/t5820/syb/OTER/configs/mnre_oter_seed568.json",
        "prototype_metric": "euclidean",
        "max_epoch": 10,
        "freeze_vision_bn": False,
        "attention_scale": 0.20,
        "ota_scale": 0.10,
        "router_scale": 0.10,
        "router_phrase_scale": 0.20,
        "phrase_scale": 0.20,
        "phrase_mode": "legacy",
        "relation_context_length": 40,
        "bert_trainable_layers": 3,
    },
    "fewrel": {
        "data_root": "/home/t5820/syb/FewREL",
        "seed": 31415,
        "eval_seed": 2026,
        "test_seed": 2027,
        "split_path": None,
        "prototype_metric": "squared_euclidean",
        "max_epoch": 10,
        "freeze_vision_bn": True,
        "attention_scale": 0.20,
        "ota_scale": 0.10,
        "router_scale": 0.10,
        "router_phrase_scale": 0.15,
        "phrase_scale": 0.20,
        "phrase_mode": "entity_context",
        "relation_context_length": 80,
        "bert_trainable_layers": 6,
    },
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the fixed Full model on MNRE or FewRel"
    )
    parser.add_argument("--dataset", choices=sorted(PROFILES), required=True)
    parser.add_argument("--data_root")
    parser.add_argument(
        "--pretrain_path", default="/home/t5820/syb/pretrained_model/bert-base-uncased"
    )
    parser.add_argument(
        "--vision_pretrained_path",
        default="/home/t5820/syb/pretrained_model/resnet50-0676ba61.pth",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--Q", type=int, default=5)
    parser.add_argument("--train_episodes", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=500)
    parser.add_argument("--max_epoch", type=int)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval_seed", type=int)
    parser.add_argument("--test_seed", type=int)
    parser.add_argument("--split_path")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--aux_loss_scale", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--init_ckpt",
        help="Optional Full checkpoint used to initialize continued training",
    )
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--only_test", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    return parser


def apply_profile(args):
    profile = dict(PROFILES[args.dataset])
    if args.dataset == "mnre":
        if (args.N, args.K) == (10, 1):
            args.full_protocol = "10w1s"
            profile.update(
                seed=123,
                max_epoch=3,
                attention_scale=0.30,
                ota_scale=0.20,
                router_scale=0.15,
                router_phrase_scale=0.30,
            )
        elif (args.N, args.K) == (5, 1):
            args.full_protocol = "5w1s"
        else:
            raise ValueError("MNRE Full package supports 5W1S and 10W1S")
    else:
        args.full_protocol = f"{args.N}w{args.K}s"
    for name in (
        "data_root", "seed", "eval_seed", "test_seed", "split_path", "max_epoch"
    ):
        if getattr(args, name) is None:
            setattr(args, name, profile[name])
    args.prototype_metric = profile["prototype_metric"]
    return profile


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_episode(batch_items, device):
    labels = batch_items[0].to(device, non_blocking=True)
    values = batch_items[1] if len(batch_items) == 2 else batch_items[2:]
    model_args = []
    for index, value in enumerate(values):
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                value = value.float()
            if len(values) == 14 and index == 9:
                batch = value.shape[0]
                value = value.reshape(batch * 3, 3, 224, 224)
            elif value.ndim == 5:
                batch, objects, channels, height, width = value.shape
                value = value.reshape(batch * objects, channels, height, width)
            value = value.to(device, non_blocking=True)
        model_args.append(value)
    if len(model_args) == 14:
        object_slots = model_args[10][:, :3]
        model_args = [
            model_args[0], model_args[1], model_args[2], model_args[3],
            model_args[4], model_args[5], model_args[7], model_args[9],
            model_args[11], model_args[12], model_args[13], object_slots,
        ]
    return labels, model_args


def prototypical_scores(labels, support, query_labels, query, metric):
    support = support.squeeze(0) if support.ndim == 3 else support
    query = query.squeeze(0) if query.ndim == 3 else query
    class_ids = torch.unique(labels, sorted=True)
    prototypes = torch.stack([support[labels == item].mean(0) for item in class_ids])
    if metric == "euclidean":
        scores = -torch.cdist(query, prototypes)
    elif metric == "squared_euclidean":
        scores = -torch.cdist(query, prototypes).square()
    else:
        raise ValueError(metric)
    lookup = {item.item(): index for index, item in enumerate(class_ids)}
    targets = torch.tensor(
        [lookup[item.item()] for item in query_labels], device=query_labels.device
    )
    return scores, targets


def mnre_paths(root):
    data = Path(root)
    return {
        "text": [
            data / "txt/ours_train.txt",
            data / "txt/ours_val.txt",
            data / "txt/ours_test.txt",
        ],
        "caption": [data / "caption/minicpm_train.txt"],
        "images": data / "img_org/train",
        "rel2id": data / "ours_rel2id.json",
        "data": data,
    }


def relation_counts(paths):
    counts = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                relation = eval(line)["relation"]
                if relation != "None":
                    counts[relation] = counts.get(relation, 0) + 1
    return counts


def build_mnre_loaders(args, encoder):
    paths = mnre_paths(args.data_root)
    split = json.loads(Path(args.split_path).read_text(encoding="utf-8"))
    rel2id = json.loads(paths["rel2id"].read_text(encoding="utf-8"))
    eligible = {
        name for name, count in relation_counts(paths["text"][:1]).items()
        if count >= args.K + args.Q
    }
    train_classes = [name for name in split["train_classes"] if name in eligible]
    eval_classes = [name for name in split["eval_classes"] if name in eligible]
    common = dict(
        text_paths=[str(path) for path in paths["text"]],
        pic_path=str(paths["images"]),
        cap_path=[str(path) for path in paths["caption"]],
        rel2id=rel2id,
        tokenizer=encoder.tokenize,
        batch_size=1,
        N=args.N,
        K=args.K,
        Q=args.Q,
        eval_episode_seed=args.eval_seed,
        eval_sampling="fixed_seed",
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        data_root=str(paths["data"]),
        pretrain_path=args.pretrain_path,
        full_image_preprocessing="oter_bgr_raw",
        num_workers=args.num_workers,
    )
    train = DynamicFewShotLoader(
        **common, shuffle=True, is_train=True, precomputed_classes=train_classes
    )
    evaluation = DynamicFewShotLoader(
        **common, shuffle=False, is_train=False, precomputed_classes=eval_classes
    )
    return train, evaluation, evaluation, rel2id


def build_rel2id(paths):
    relations = set()
    for path in paths:
        relations.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return {relation: index for index, relation in enumerate(sorted(relations))}


def build_fewrel_loaders(args, encoder):
    root = Path(args.data_root)
    train_path = root / "train_small.json"
    val_path = root / "val_small.json"
    rel2id_path = root / "rel2id.json"
    rel2id = (
        json.loads(rel2id_path.read_text(encoding="utf-8"))
        if rel2id_path.exists() else build_rel2id([train_path, val_path])
    )
    missing = sorted(set(build_rel2id([train_path, val_path])) - set(rel2id))
    next_id = max(rel2id.values(), default=-1) + 1
    rel2id.update({relation: next_id + i for i, relation in enumerate(missing)})
    train_data = json.loads(train_path.read_text(encoding="utf-8"))
    active = [
        relation for relation, items in train_data.items()
        if len(items) >= args.K + args.Q
    ]
    common = dict(
        pic_path=str(root / "image"),
        rel2id=rel2id,
        tokenizer=encoder.tokenize,
        batch_size=1,
        K=args.K,
        Q=args.Q,
        caption_path=str(root / "fewrel_caption_dedup_v1.txt"),
        caption_mode="entity_scene",
        obj_path=str(root / "output/crops_mixed"),
        obj_manifest_path=str(root / "output/crops_mixed_manifest.jsonl"),
        entity_obj_metadata_path=str(root / "output/crops_text_boxes.jsonl"),
        bert_tokenizer=encoder.tokenizer,
        max_length=args.max_length,
        image_augmentation="none",
        num_workers=args.num_workers,
    )
    train = FewRELFewShotLoader(
        **common,
        data_paths=[str(train_path)],
        shuffle=True,
        N=args.N,
        is_train=True,
        active_classes=active,
        supplementary_text_path=str(root / "train_wiki_official.json"),
        supplementary_queries_per_class=min(5, args.Q),
        supplementary_visual_mode="none",
        episodes=args.train_episodes,
    )
    evaluation = FewRELFewShotLoader(
        **common,
        data_paths=[str(val_path)],
        shuffle=False,
        N=args.N,
        is_train=False,
        active_classes=None,
        episodes=args.eval_episodes,
    )
    return train, evaluation, evaluation, rel2id


def evaluate(model, loader, args, episode_seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    set_seed(episode_seed)
    model.eval()
    accuracies = []
    try:
        with torch.no_grad():
            for support, query in tqdm(loader, desc="Evaluation", unit="ep"):
                support_labels, support_args = prepare_episode(support, args.device)
                query_labels, query_args = prepare_episode(query, args.device)
                _, support_rep, _ = model(*support_args)
                _, query_rep, _ = model(*query_args)
                scores, targets = prototypical_scores(
                    support_labels, support_rep, query_labels, query_rep,
                    args.prototype_metric,
                )
                accuracies.append(
                    (scores.argmax(-1) == targets).float().mean().item()
                )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    accuracy = 100.0 * float(np.mean(accuracies))
    ci95 = 0.0
    if len(accuracies) > 1:
        ci95 = 100.0 * 1.96 * float(np.std(accuracies, ddof=1)) / math.sqrt(
            len(accuracies)
        )
    return accuracy, ci95


def train_epoch(model, loader, optimizer, args, epoch):
    model.train()
    losses, accuracies = [], []
    progress = tqdm(loader, desc=f"Epoch {epoch}/{args.max_epoch}", unit="ep")
    for support, query in progress:
        support_labels, support_args = prepare_episode(support, args.device)
        query_labels, query_args = prepare_episode(query, args.device)
        optimizer.zero_grad(set_to_none=True)
        _, support_rep, support_aux = model(*support_args)
        _, query_rep, query_aux = model(*query_args)
        scores, targets = prototypical_scores(
            support_labels, support_rep, query_labels, query_rep,
            args.prototype_metric,
        )
        auxiliary = 0.5 * (support_aux.mean() + query_aux.mean())
        loss = F.cross_entropy(scores, targets) + args.aux_loss_scale * auxiliary
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        accuracy = (scores.argmax(-1) == targets).float().mean().item()
        losses.append(loss.item())
        accuracies.append(accuracy)
        progress.set_postfix(loss=f"{loss.item():.3f}", acc=f"{accuracy*100:.1f}")
    return float(np.mean(losses)), 100.0 * float(np.mean(accuracies))


def checkpoint_state(model, args, epoch, accuracy, ci95):
    return {
        "state_dict": model.state_dict(),
        "dataset": args.dataset,
        "epoch": epoch,
        "accuracy": accuracy,
        "ci95": ci95,
        "protocol": {"N": args.N, "K": args.K, "Q": args.Q},
    }


def main():
    args = build_parser().parse_args()
    profile = apply_profile(args)
    if args.smoke_test:
        args.N, args.K, args.Q = 2, 1, 1
        args.train_episodes = args.eval_episodes = args.max_epoch = 1
        args.num_workers = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    set_seed(args.seed)
    vision_path = args.vision_pretrained_path
    if vision_path and not Path(vision_path).exists():
        if args.only_test or args.init_ckpt:
            LOGGER.warning(
                "ImageNet ResNet file is absent; checkpoint weights will replace it"
            )
            vision_path = None
        else:
            raise FileNotFoundError(
                "Fresh training requires --vision_pretrained_path: " + vision_path
            )
    encoder = FullOTER(
        max_length=args.max_length,
        pretrain_path=args.pretrain_path,
        vision_pretrained_path=vision_path,
        freeze_vision_bn=profile["freeze_vision_bn"],
        attention_residual_scale=profile["attention_scale"],
        ota_residual_scale=profile["ota_scale"],
        text_router_residual_scale=profile["router_scale"],
        text_evidence_loss_weight=0.03,
        text_router_temperature=0.7,
        text_router_margin=0.1,
        text_router_phrase_scale=profile["router_phrase_scale"],
        phrase_feature_scale=profile["phrase_scale"],
        text_feature_scale=0.85,
        visual_feature_scale=0.90,
        phrase_mode=profile["phrase_mode"],
        phrase_max_length=24,
        relation_context_length=profile["relation_context_length"],
        bert_trainable_layers=profile["bert_trainable_layers"],
    )
    builders = {"mnre": build_mnre_loaders, "fewrel": build_fewrel_loaders}
    train_loader, val_loader, test_loader, rel2id = builders[args.dataset](
        args, encoder
    )
    model = FullRelationModel(encoder, len(rel2id), rel2id).to(args.device)
    if args.init_ckpt:
        initialization = torch.load(
            args.init_ckpt, map_location=args.device, weights_only=False
        )
        model.load_state_dict(initialization["state_dict"], strict=True)
        LOGGER.info("Strictly initialized from %s", args.init_ckpt)
    total = sum(item.numel() for item in model.parameters())
    trainable = sum(item.numel() for item in model.parameters() if item.requires_grad)
    LOGGER.info(
        "dataset=%s protocol=%s Full=OTA+paired-attention+EAT parameters=%d trainable=%d",
        args.dataset, args.full_protocol, total, trainable,
    )
    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    validation = []
    if not args.only_test:
        optimizer = torch.optim.AdamW(
            (item for item in model.parameters() if item.requires_grad),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        best_accuracy = -1.0
        for epoch in range(1, args.max_epoch + 1):
            train_loss, train_accuracy = train_epoch(
                model, train_loader, optimizer, args, epoch
            )
            val_accuracy, val_ci95 = evaluate(
                model, val_loader, args, args.eval_seed
            )
            validation.append(
                {"epoch": epoch, "accuracy": val_accuracy, "ci95": val_ci95}
            )
            LOGGER.info(
                "epoch=%d loss=%.4f train=%.2f val=%.2f +/- %.2f",
                epoch, train_loss, train_accuracy, val_accuracy, val_ci95,
            )
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(
                    checkpoint_state(model, args, epoch, val_accuracy, val_ci95),
                    ckpt_path,
                )
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test_accuracy, test_ci95 = evaluate(model, test_loader, args, args.test_seed)
    metrics = {
        "model": "Full (OTA + paired attention + EAT)",
        "dataset": args.dataset,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "test_seed": args.test_seed,
        "protocol": {"N": args.N, "K": args.K, "Q": args.Q},
        "full_topology": args.full_protocol,
        "validation": validation,
        "test": {"accuracy": test_accuracy, "ci95": test_ci95},
        "checkpoint": str(ckpt_path.resolve()),
    }
    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Final accuracy: %.2f +/- %.2f", test_accuracy, test_ci95)


if __name__ == "__main__":
    main()
