import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import cv2
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms

class DynamicSplitFewShotDataset(data.Dataset):
    def __init__(self, text_paths, pic_path, cap_path, rel2id, tokenizer, N, K, Q, is_train=True, train_class_ratio=0.7, precomputed_classes=None, kwargs=None):
        super().__init__()
        self.kwargs = kwargs if kwargs is not None else {}
        for intercept_key in ['train_class_ratio', 'is_train', 'N', 'K', 'Q', 'max_length', 'class_seed', 'precomputed_classes']:
            if intercept_key in self.kwargs:
                self.kwargs.pop(intercept_key)
        self.pic_path = pic_path
        self.N = N
        self.K = K
        self.Q = Q
        self.rel2id = rel2id
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.eval_episode_seed = self.kwargs.pop('eval_episode_seed', 3407)
        self.eval_sampling = self.kwargs.pop('eval_sampling', 'fixed_seed')
        if self.eval_sampling not in ('fixed_seed', 'legacy_random'):
            raise ValueError(
                'eval_sampling must be fixed_seed or legacy_random, got '
                f'{self.eval_sampling}'
            )
        self.train_episodes = self.kwargs.pop('train_episodes', 1000)
        self.eval_episodes = self.kwargs.pop('eval_episodes', 500)
        self.data_root = self.kwargs.pop(
            'data_root', '/home/t5820/syb/data'
        )
        self.pretrain_path = self.kwargs.pop(
            'pretrain_path', '/home/t5820/syb/pretrained_model/bert-base-uncased'
        )
        self.full_image_preprocessing = self.kwargs.pop(
            'full_image_preprocessing', 'imagenet_rgb'
        )
        if self.full_image_preprocessing not in ('imagenet_rgb', 'oter_bgr_raw'):
            raise ValueError(
                'full_image_preprocessing must be imagenet_rgb or '
                f'oter_bgr_raw, got {self.full_image_preprocessing}'
            )
        self.kwargs.pop('num_workers', None)

        self.all_json_data = {}
        self.item_mode_map = {}
        self.item_index_map = {}

        for text_path in text_paths:
            if not os.path.exists(text_path):
                continue
            if 'train' in text_path:
                mode = 'train'
            elif 'val' in text_path:
                mode = 'val'
            else:
                mode = 'test'

            with open(text_path, encoding='UTF-8') as f:
                line_idx = 0
                for line in f:
                    line = line.rstrip()
                    if line:
                        item = eval(line)
                        rel = item['relation']
                        if rel == 'None':
                            line_idx += 1
                            continue
                        if rel not in self.all_json_data:
                            self.all_json_data[rel] = []
                        self.all_json_data[rel].append(item)
                        self.item_mode_map[item['img_id']] = mode
                        self.item_index_map[item['img_id']] = line_idx
                        line_idx += 1

        qualified_classes = [r for r, inst in self.all_json_data.items() if len(inst) >= (self.K + self.Q)]

        if precomputed_classes is not None:
            self.active_classes = [c for c in precomputed_classes if c in qualified_classes]
            
            dropped = set(precomputed_classes) - set(self.active_classes)
            if dropped:
                logging.warning(
                    f"[外部大类联动过滤] 以下类别虽在预设列表中，但因当前文本样本数不足 K+Q={self.K + self.Q} 而被直接筛除: {list(dropped)}"
                )
        else:
            random.shuffle(qualified_classes)
            num_train_classes = int(len(qualified_classes) * train_class_ratio)
            self.active_classes = qualified_classes[:num_train_classes] if self.is_train else qualified_classes[num_train_classes:]

        logging.debug(f"==== 动态少样本大类安全过滤成功 ({'训练模式' if self.is_train else '测试模式'}) ====")
        logging.debug(f"当前文本总类别数: {len(self.all_json_data)} | 严格满足无重复采样的激活关系数: {len(self.active_classes)}")

        self.cap_dict = {}
        cap_paths = cap_path if isinstance(cap_path, (list, tuple)) else [cap_path]
        for single_cap_path in cap_paths:
            if os.path.exists(single_cap_path):
                with open(single_cap_path, encoding='UTF-8') as f_cap:
                    for line in f_cap:
                        line = line.rstrip()
                        if line:
                            item = eval(line)
                            self.cap_dict[item['img_id']] = item['text']

        self.phrase_data = {}
        self.state_dict_ori = {}

        data_root = os.path.join(self.data_root, 'txt')
        for m in ['train', 'val', 'test']:
            p_path = os.path.join(data_root, f'phrase_text_{m}.json')
            if os.path.exists(p_path):
                with open(p_path, 'r') as f_g:
                    self.phrase_data[m] = json.load(f_g)
            else:
                self.phrase_data[m] = {}

            ori_p = os.path.join(data_root, f'mre_{m}_dict.pth')
            if os.path.exists(ori_p):
                self.state_dict_ori[m] = torch.load(ori_p)
            else:
                self.state_dict_ori[m] = {}

        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.tokenizer_cap = BertTokenizer.from_pretrained(self.pretrain_path)
        self.cached_eval_episodes = (
            self._build_eval_episodes()
            if not self.is_train and self.eval_sampling == 'fixed_seed'
            else None
        )

    def __len__(self):
        return self.train_episodes if self.is_train else self.eval_episodes

    def _sample_episode_plan(self, rng):
        current_N = min(self.N, len(self.active_classes))

        if current_N == 0:
            raise ValueError(f"[严重错误] 当前 Dataset 没有任何一个关系的样本数能满足 K+Q={self.K+self.Q} 的少样本任务！")

        target_classes = rng.sample(self.active_classes, current_N)
        episode_plan = []
        for class_name in target_classes:
            all_instances = self.all_json_data[class_name]
            sampled_indices = rng.sample(range(len(all_instances)), self.K + self.Q)
            episode_plan.append((class_name, sampled_indices))
        return episode_plan

    def _build_eval_episodes(self):
        rng = random.Random(self.eval_episode_seed)
        return [self._sample_episode_plan(rng) for _ in range(self.__len__())]

    def _process_single_instance(self, item):
        img_id = item['img_id']

        mode = self.item_mode_map.get(img_id, 'train')
        line_idx = self.item_index_map.get(img_id, None)

        aux_imgs_ori = []
        # The episodic tensor is padded to three crops. Keep an explicit mask
        # so an absent crop is not later interpreted as visual evidence.
        object_slot_mask = torch.zeros(3, dtype=torch.float)
        for _ in range(3): aux_imgs_ori.append(torch.zeros((3, 224, 224)))

        actual_ori_path = None
        for folder_mode in ['train', 'val', 'test']:
            base_dir = self.pic_path.rstrip('/').replace('img_org/train', 'img_org').replace('img_org', f'img_org/{folder_mode}')
            potential_path = os.path.join(base_dir, img_id)
            if os.path.exists(potential_path):
                actual_ori_path = potential_path
                break
        if actual_ori_path is None:
            actual_ori_path = os.path.join(self.pic_path, img_id)

        pic_path_coarse_ori = self.pic_path.replace('org', 'vg')

        if line_idx is not None:
            state_dict_ori_mode = self.state_dict_ori.get(mode, {})
            if line_idx in state_dict_ori_mode:
                crops_list = state_dict_ori_mode[line_idx]
                for i in range(min(3, len(crops_list))):
                    try:
                        crop_name = crops_list[i]
                        crop_path = os.path.join(pic_path_coarse_ori, 'crops', crop_name)
                        if os.path.exists(crop_path):
                            image_ori_object = Image.open(crop_path).convert('RGB')
                            aux_imgs_ori[i] = self.transform(image_ori_object)
                            object_slot_mask[i] = 1.0
                    except:
                        pass

            phrase_data_mode = self.phrase_data.get(mode, {})
            item['grounding'] = phrase_data_mode.get(str(line_idx), "object")
        else:
            item['grounding'] = "object"
        seq = list(self.tokenizer(item, **self.kwargs))

        cap_text = self.cap_dict.get(img_id, "photo")
        enc_cap = self.tokenizer_cap.encode_plus(text=cap_text, max_length=40, truncation=True, padding='max_length')
        cap_ids = torch.tensor(enc_cap['input_ids'])
        cap_types = torch.tensor(enc_cap['token_type_ids'])
        cap_masks = torch.tensor(enc_cap['attention_mask'])

        t_ori = torch.zeros((3, 224, 224))
        try:
            if actual_ori_path and os.path.exists(actual_ori_path):
                if self.full_image_preprocessing == 'oter_bgr_raw':
                    # Original CAMIM fed the OpenCV BGR image directly to the
                    # visual encoder. Keep this only as an explicit protocol
                    # compatibility mode; the MNRE3 default uses ImageNet RGB.
                    image_bgr = cv2.imread(actual_ori_path)
                    if image_bgr is not None:
                        image_bgr = cv2.resize(
                            image_bgr, (224, 224), interpolation=cv2.INTER_AREA
                        )
                        t_ori = torch.from_numpy(image_bgr).permute(2, 0, 1).float()
                else:
                    t_ori = self.transform(
                        Image.open(actual_ori_path).convert('RGB')
                    )
        except Exception:
            pass

        return {
            'relation': self.rel2id[item['relation']],
            'img_id': img_id,
            'indexed_tokens': seq[0].squeeze(0),
            'att_mask': seq[1].squeeze(0),
            'pos1': seq[2].squeeze(0).squeeze(0), 
            'pos2': seq[3].squeeze(0).squeeze(0),
            'token_phrases': seq[4].squeeze(0),
            'att_mask_phrases': seq[5].squeeze(0),
            't_ori': t_ori,
            'aux_imgs_ori': torch.stack(aux_imgs_ori, dim=0), 
            'object_slot_mask': object_slot_mask,
            'cap_ids': cap_ids,
            'cap_types': cap_types,
            'cap_masks': cap_masks
        }

    def __getitem__(self, index):
        support_set, query_set = [], []
        if self.is_train:
            episode_plan = self._sample_episode_plan(random)
        else:
            if self.eval_sampling == 'legacy_random':
                episode_plan = self._sample_episode_plan(random)
            else:
                episode_plan = self.cached_eval_episodes[index]

        for class_name, sampled_indices in episode_plan:
            all_instances = self.all_json_data[class_name]
            sampled_instances = [all_instances[idx] for idx in sampled_indices]
            support_instances = sampled_instances[:self.K]
            query_instances = sampled_instances[self.K:]

            for item in support_instances:
                support_set.append(self._process_single_instance(item))
            for item in query_instances:
                query_set.append(self._process_single_instance(item))

        return support_set, query_set

    @staticmethod
    def collate_fn(batch):
        support_set, query_set = batch[0]

        def transpose_and_tensorize(dataset_set):
            labels = torch.tensor([item['relation'] for item in dataset_set]).long()
            img_ids = [item['img_id'] for item in dataset_set]

            indexed_tokens = torch.stack([item['indexed_tokens'] for item in dataset_set], dim=0)
            att_mask = torch.stack([item['att_mask'] for item in dataset_set], dim=0)
            pos1 = torch.stack([item['pos1'] for item in dataset_set], dim=0).unsqueeze(-1)
            pos2 = torch.stack([item['pos2'] for item in dataset_set], dim=0).unsqueeze(-1)
            token_phrases = torch.stack([item['token_phrases'] for item in dataset_set], dim=0)
            att_mask_phrases = torch.stack([item['att_mask_phrases'] for item in dataset_set], dim=0)

            t_ori = torch.stack([item['t_ori'] for item in dataset_set], dim=0)

            aux_imgs_ori = torch.stack([item['aux_imgs_ori'] for item in dataset_set], dim=0)
            object_slot_mask = torch.stack(
                [item['object_slot_mask'] for item in dataset_set], dim=0
            )

            cap_ids = torch.stack([item['cap_ids'] for item in dataset_set], dim=0)
            cap_types = torch.stack([item['cap_types'] for item in dataset_set], dim=0)
            cap_masks = torch.stack([item['cap_masks'] for item in dataset_set], dim=0)

            return [labels, img_ids, indexed_tokens, att_mask, pos1, pos2,
                    token_phrases, att_mask_phrases, t_ori, aux_imgs_ori,
                    cap_ids, cap_types,
                    cap_masks, object_slot_mask]

        return transpose_and_tensorize(support_set), transpose_and_tensorize(query_set)


def DynamicFewShotLoader(text_paths, pic_path, cap_path, rel2id, tokenizer, batch_size, shuffle, N, K, Q, is_train=True, precomputed_classes=None, **kwargs):
    # Dataset initialization consumes its private options with ``pop``. Keep
    # the loader-level worker configuration outside that mutable copy.
    num_workers = kwargs.get('num_workers', 4 if is_train else 0)
    dataset = DynamicSplitFewShotDataset(text_paths=text_paths, pic_path=pic_path, cap_path=cap_path,
                                         rel2id=rel2id, tokenizer=tokenizer, N=N, K=K, Q=Q, is_train=is_train,
                                         train_class_ratio=kwargs.get('train_class_ratio', 0.7),
                                         precomputed_classes=precomputed_classes,
                                         kwargs=dict(kwargs))
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers, collate_fn=DynamicSplitFewShotDataset.collate_fn)
