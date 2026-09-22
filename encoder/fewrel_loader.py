import torch
import torch.utils.data as data
import os, random, json, logging, re, unicodedata
from collections import Counter
from torchvision import transforms
from PIL import Image


class FewRELFewShotDataset(data.Dataset):

    def __init__(
        self,
        data_paths,
        pic_path,
        rel2id,
        tokenizer,
        N, K, Q,
        is_train=True,
        max_length=128,
        active_classes=None,
        caption_path=None,
        caption_mode='entity_scene',
        obj_path=None,
        obj_manifest_path=None,
        entity_obj_metadata_path=None,
        supplementary_text_path=None,
        supplementary_queries_per_class=0,
        supplementary_visual_mode='none',
        bert_tokenizer=None,
        image_augmentation='none',
        episodes=None,
        kwargs=None
    ):
        super().__init__()

        self.tokenizer_kwargs = kwargs if kwargs is not None else {}
        _drop_keys = ['train_class_ratio', 'is_train', 'N', 'K', 'Q',
                      'max_length', 'class_seed',
                      'image_augmentation']
        for k in _drop_keys:
            self.tokenizer_kwargs.pop(k, None)

        self.pic_path = pic_path
        self.obj_path = obj_path
        self.obj_manifest = {}
        self.entity_obj_metadata = {}
        self.N = N
        self.K = K
        self.Q = Q
        self.rel2id = rel2id
        self.tokenizer = tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.is_train = is_train
        if caption_mode not in {
            'entity_scene', 'entity_only', 'entity_filtered_scene',
            'marked_entity_scene',
            'marked_entity_detection_scene',
        }:
            raise ValueError(
                'caption_mode must be one of: entity_scene, entity_only, '
                'entity_filtered_scene, marked_entity_scene, '
                'marked_entity_detection_scene'
            )
        self.caption_mode = caption_mode
        self.supplementary_queries_per_class = supplementary_queries_per_class
        if supplementary_visual_mode not in {'none', 'same_relation'}:
            raise ValueError(
                'supplementary_visual_mode must be none or same_relation'
            )
        self.supplementary_visual_mode = supplementary_visual_mode
        if not 0 <= supplementary_queries_per_class <= Q:
            raise ValueError(
                'supplementary_queries_per_class must be in [0, Q]'
            )
        if supplementary_queries_per_class and not is_train:
            raise ValueError(
                'supplementary text samples are only allowed in the train loader'
            )
        if image_augmentation not in ('none', 'color_jitter'):
            raise ValueError(
                'image_augmentation must be one of: none, color_jitter'
            )
        self.image_augmentation = image_augmentation
        self.max_length = max_length
        self.episodes = episodes if episodes is not None else (1000 if is_train else 500)
        if self.episodes <= 0:
            raise ValueError("episodes must be a positive integer")

        # Mixed-crop manifests preserve information that cannot be recovered
        # from the sequential filenames alone (text-grounded vs. caption-
        # grounded source and per-slot quality).  Loading it is harmless for
        # legacy variants because the resulting weights are only consumed by
        # source-aware model configurations.
        if obj_manifest_path is None and obj_path:
            normalized_obj_path = os.path.normpath(obj_path)
            inferred_manifest = os.path.join(
                os.path.dirname(normalized_obj_path),
                f"{os.path.basename(normalized_obj_path)}_manifest.jsonl"
            )
            if os.path.exists(inferred_manifest):
                obj_manifest_path = inferred_manifest
        if obj_manifest_path and os.path.exists(obj_manifest_path):
            with open(obj_manifest_path, encoding='UTF-8') as manifest_file:
                for line in manifest_file:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self.obj_manifest[record['img_id']] = record.get('crops', [])
            logging.info(
                "加载 obj manifest 完成: %s, 共 %d 条记录",
                obj_manifest_path,
                len(self.obj_manifest),
            )

        # Optional per-detection grounding phrases make crop selection
        # instance-aware.  This matters when one image ID is reused by
        # multiple FewRel entity pairs: the image-level mixed directory may
        # otherwise feed a crop grounded by another sample's entities.
        if entity_obj_metadata_path:
            if not os.path.exists(entity_obj_metadata_path):
                raise FileNotFoundError(entity_obj_metadata_path)
            with open(entity_obj_metadata_path, encoding='UTF-8') as metadata_file:
                for line in metadata_file:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get('img_id'):
                        # The extractor is append-only; the final record is the
                        # most recent extraction for this image.
                        self.entity_obj_metadata[record['img_id']] = record
            logging.info(
                "加载 entity-level obj metadata 完成: %s, 共 %d 条记录",
                entity_obj_metadata_path,
                len(self.entity_obj_metadata),
            )

        # ---- 加载 caption 文件 ----
        self.caption_dict = {}
        if caption_path and os.path.exists(caption_path):
            with open(caption_path, encoding='UTF-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        img_name, caption_text = parts
                        img_id_key = os.path.splitext(img_name)[0]
                        self.caption_dict[img_id_key] = caption_text
            logging.info(f"加载 caption 文件完成, 共 {len(self.caption_dict)} 条记录")
        else:
            logging.warning(f"caption 文件未找到或未指定: {caption_path}")

        # Shared exact captions tend to be generic templates rather than
        # image-specific evidence.  The filtered mode below keeps unique
        # scene descriptions while dropping only captions reused by multiple
        # image IDs; it never modifies the source caption file.
        self.caption_frequency = Counter(
            re.sub(r'\s+', ' ', caption.strip().lower())
            for caption in self.caption_dict.values()
            if caption.strip()
        )

        self.all_json_data = {}
        for path in data_paths:
            if not os.path.exists(path):
                logging.warning(f"数据文件不存在，跳过: {path}")
                continue
            with open(path, encoding='UTF-8') as f:
                raw = json.load(f)
            for rel, instances in raw.items():
                if rel not in self.all_json_data:
                    self.all_json_data[rel] = []
                self.all_json_data[rel].extend(instances)

        # ---- 过滤掉不在 rel2id 中的关系 ----
        valid_rels = [r for r in self.all_json_data if r in self.rel2id]
        if len(valid_rels) < len(self.all_json_data):
            dropped = set(self.all_json_data.keys()) - set(valid_rels)
            logging.warning(f"以下关系不在 rel2id 中，已跳过: {dropped}")
        self.all_json_data = {r: self.all_json_data[r] for r in valid_rels}

        self.supplementary_text_data = {}
        if supplementary_queries_per_class:
            if not supplementary_text_path or not os.path.exists(
                supplementary_text_path
            ):
                raise FileNotFoundError(supplementary_text_path)
            with open(supplementary_text_path, encoding='UTF-8') as handle:
                raw_supplement = json.load(handle)
            for relation, instances in self.all_json_data.items():
                existing_sentences = {
                    tuple(item.get('tokens', [])) for item in instances
                }
                candidates = [
                    item for item in raw_supplement.get(relation, [])
                    if tuple(item.get('tokens', [])) not in existing_sentences
                ]
                if candidates:
                    self.supplementary_text_data[relation] = candidates
            missing_supplement = sorted(
                set(self.all_json_data) - set(self.supplementary_text_data)
            )
            if missing_supplement:
                raise ValueError(
                    'No non-overlapping supplementary instances for: '
                    + ', '.join(missing_supplement)
                )
            logging.info(
                "训练文本补充池: %s, 共 %d 条非重合实例; 每类 query 替换 %d/%d",
                supplementary_text_path,
                sum(map(len, self.supplementary_text_data.values())),
                supplementary_queries_per_class,
                Q,
            )

        # ---- 设置激活类别 ----
        if active_classes is not None:
            self.active_classes = [c for c in active_classes if c in self.all_json_data]
            dropped = set(active_classes) - set(self.active_classes)
            if dropped:
                logging.warning(f"active_classes 中以下类别在数据里找不到，已跳过: {dropped}")
        else:
            self.active_classes = list(self.all_json_data.keys())
            logging.info("未传入 active_classes, 使用数据文件中全部类别")

        logging.info(f"==== FewREL Dataset ({'训练' if is_train else '测试'}) ====")
        logging.info(f"当前激活类别数: {len(self.active_classes)}")
        logging.info(f"当前激活类别: {sorted(self.active_classes)}")

        if self.obj_path:
            dataset_image_ids = {
                item.get("img_id", "")
                for instances in self.all_json_data.values()
                for item in instances
                if item.get("img_id", "")
            }
            crop_counts = Counter()
            for image_id in dataset_image_ids:
                crop_counts[sum(
                    os.path.exists(os.path.join(
                        self.obj_path,
                        f"{image_id}_pred_yolo_crop_{index:05d}.png"
                    ))
                    for index in range(3)
                )] += 1
            covered = len(dataset_image_ids) - crop_counts.get(0, 0)
            logging.info(
                "obj 覆盖: %d/%d (%.1f%%), crop 数分布=%s",
                covered,
                len(dataset_image_ids),
                100.0 * covered / max(1, len(dataset_image_ids)),
                dict(sorted(crop_counts.items())),
            )
            if self.obj_manifest:
                source_patterns = Counter(
                    tuple(
                        crop.get('source', 'unknown')
                        for crop in self.obj_manifest.get(image_id, [])[:3]
                    )
                    for image_id in dataset_image_ids
                )
                logging.info(
                    "obj 来源组合分布: %s",
                    dict(sorted(
                        source_patterns.items(), key=lambda item: str(item[0])
                    )),
                )

        transform_ops = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        if self.is_train and self.image_augmentation == 'color_jitter':
            # Keep geometric alignment stable for text-grounded crops; only
            # perturb appearance to regularize the small visual training set.
            transform_ops.append(
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.10,
                )
            )
        transform_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform = transforms.Compose(transform_ops)

    @staticmethod
    def _to_mnre_format(item):
        def parse_pos(pos_data):
            segment = pos_data[0]
            start = segment[0]
            end = segment[-1] + 1
            return [start, end]

        return {
            "token": item["tokens"],
            "h": {"name": item["h"][0], "pos": parse_pos(item["h"][2])},
            "t": {"name": item["t"][0], "pos": parse_pos(item["t"][2])},
            "img_id": item.get("img_id", ""),
            "relation": item.get("relation", ""),
            "grounding": " ".join(item["tokens"])
        }

    def _matching_detection_phrases(self, img_id, h_name, t_name):
        """Return de-duplicated detection phrases grounded to this entity pair."""
        phrases = []
        normalized_phrases = set()
        for detection in self.entity_obj_metadata.get(img_id, {}).get(
            'detections', []
        ):
            phrase = str(detection.get('phrase', '')).strip()
            if not phrase:
                continue
            if (
                self._phrase_matches_entity(phrase, h_name)
                or self._phrase_matches_entity(phrase, t_name)
            ):
                normalized = self._normalize_entity_phrase(phrase)
                if normalized and normalized not in normalized_phrases:
                    phrases.append(phrase)
                    normalized_phrases.add(normalized)
        return phrases

    def _build_entity_caption(self, img_id, h_name, t_name):
        scene_caption = self.caption_dict.get(img_id, "")
        first_sent = scene_caption.split('.')[0].strip() if scene_caption else ''
        if self.caption_mode in {'entity_scene', 'entity_filtered_scene'}:
            prefix = f"{h_name} and {t_name}"
        else:
            # Reuse the relation encoder's reserved markers so the caption
            # path explicitly retains head/tail roles and relation direction.
            prefix = (
                f"[unused0] {h_name} [unused1] "
                f"[unused2] {t_name} [unused3]"
            )
        segments = [prefix]
        if self.caption_mode == 'marked_entity_detection_scene':
            detection_phrases = self._matching_detection_phrases(
                img_id, h_name, t_name
            )
            if detection_phrases:
                segments.append(
                    'grounded objects: ' + ' ; '.join(detection_phrases)
                )
        # Generic visual captions are often shared by many unrelated FewRel
        # images. The legacy entity-only input mode keeps the caption encoder
        # as the entity-description text modality while excluding that noisy
        # scene sentence; the Full package itself uses entity_scene.
        normalized_scene = re.sub(r'\s+', ' ', scene_caption.strip().lower())
        scene_is_shared = self.caption_frequency.get(normalized_scene, 0) > 1
        if (
            first_sent
            and self.caption_mode != 'entity_only'
            and not (
                self.caption_mode == 'entity_filtered_scene'
                and scene_is_shared
            )
        ):
            segments.append(first_sent)
        return ' . '.join(segments)

    def _tokenize_caption(self, img_id, h_name, t_name,
                          fallback_token_ids, fallback_att_mask):
        if self.bert_tokenizer is not None:
            caption_text = self._build_entity_caption(img_id, h_name, t_name)
            encoding = self.bert_tokenizer(
                caption_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            caption_ids      = encoding['input_ids'].squeeze(0)
            caption_type_ids = encoding['token_type_ids'].squeeze(0)
            caption_att_mask = encoding['attention_mask'].squeeze(0)
        else:
            caption_ids      = fallback_token_ids.clone()
            caption_type_ids = torch.zeros_like(fallback_token_ids)
            caption_att_mask = fallback_att_mask.clone()

        return caption_ids, caption_type_ids, caption_att_mask

    @staticmethod
    def _normalize_entity_phrase(value):
        ascii_value = unicodedata.normalize(
            'NFKD', str(value)
        ).encode('ascii', 'ignore').decode('ascii')
        return " ".join(re.findall(r"[a-z0-9]+", ascii_value.lower()))

    @classmethod
    def _phrase_matches_entity(cls, phrase, entity):
        phrase = cls._normalize_entity_phrase(phrase)
        entity = cls._normalize_entity_phrase(entity)
        return bool(phrase and entity and (phrase in entity or entity in phrase))

    def _entity_matched_slots(self, img_id, loaded_slots, h_name, t_name):
        """Return relation-specific mixed slots, or all slots as a safe fallback."""
        detections = self.entity_obj_metadata.get(img_id, {}).get('detections', [])
        if not detections:
            return list(loaded_slots)
        manifest_by_slot = {
            int(crop.get('slot', -1)): crop
            for crop in self.obj_manifest.get(img_id, [])
        }
        matched_text_slots = []
        neutral_slots = []
        for slot in loaded_slots:
            metadata = manifest_by_slot.get(slot, {})
            if metadata.get('source') != 'text':
                neutral_slots.append(slot)
                continue
            source_name = os.path.basename(metadata.get('source_path', ''))
            match = re.search(r'_crop_(\d+)\.png$', source_name)
            source_index = int(match.group(1)) if match else slot
            phrase = (
                detections[source_index].get('phrase', '')
                if source_index < len(detections) else ''
            )
            if (
                self._phrase_matches_entity(phrase, h_name)
                or self._phrase_matches_entity(phrase, t_name)
            ):
                matched_text_slots.append(slot)
        if not matched_text_slots:
            return list(loaded_slots)
        return (matched_text_slots + neutral_slots)[:3]

    def _load_obj_images(self, img_id, img_ori_tensor, h_name='', t_name=''):
        MAX_OBJ = 3
        loaded_crops = []
        loaded_slots = []

        if self.obj_path:
            for idx in range(MAX_OBJ):
                crop_filename = f"{img_id}_pred_yolo_crop_{idx:05d}.png"
                crop_filepath = os.path.join(self.obj_path, crop_filename)
                if os.path.exists(crop_filepath):
                    try:
                        img = Image.open(crop_filepath).convert('RGB')
                        loaded_crops.append(self.transform(img))  # (3,224,224)
                        loaded_slots.append(idx)
                    except Exception as e:
                        logging.warning(f"obj 图像加载失败: {crop_filepath}, 原因: {e}")

        if loaded_crops and self.entity_obj_metadata:
            selected_slots = self._entity_matched_slots(
                img_id, loaded_slots, h_name, t_name
            )
            selected = [
                (crop, slot) for crop, slot in zip(loaded_crops, loaded_slots)
                if slot in selected_slots
            ]
            loaded_crops = [crop for crop, _ in selected]
            loaded_slots = [slot for _, slot in selected]

        if len(loaded_crops) == 0:
            filled = [img_ori_tensor.clone() for _ in range(MAX_OBJ)]
            # Slot 0 represents the whole-image fallback; the remaining two
            # are padding rather than three independent observations.
            slot_weights = [0.5, 0.0, 0.0]
        else:
            n = len(loaded_crops)
            filled = [loaded_crops[i % n] for i in range(MAX_OBJ)]

            manifest_by_slot = {
                int(crop.get('slot', -1)): crop
                for crop in self.obj_manifest.get(img_id, [])
            }
            slot_weights = []
            caption_count = 0
            for slot in loaded_slots:
                metadata = manifest_by_slot.get(slot, {})
                source = metadata.get('source')
                caption_count += int(source == 'caption')
                # Text-grounded crops directly target the relation entities;
                # caption crops retain useful context but receive a gentler
                # prior because their grounding phrase is scene-level.
                slot_weights.append(0.5 if source == 'caption' else 1.0)
            slot_weights.extend([0.0] * (MAX_OBJ - len(slot_weights)))

        if len(loaded_crops) == 0:
            caption_fraction = 0.0
        else:
            caption_fraction = caption_count / len(loaded_crops)

        # The first three values are OTA object-slot reliabilities and the
        # fourth is retained for backwards-compatible crop-source metadata.
        # The final flag distinguishes a genuinely text-only FewRel record
        # from a visual record whose object detector found no crop.
        object_weights = torch.tensor(
            slot_weights + [caption_fraction, 1.0], dtype=torch.float32
        )

        return torch.cat(filled, dim=0), object_weights  # (9, 224, 224), (4,)

    def _process_single_instance(self, item, rel_name):
        mnre_item = self._to_mnre_format(item)
        t = self.tokenizer(mnre_item, **self.tokenizer_kwargs)

        # In same-relation multimodal mixup, the relation sentence remains
        # the official supplementary text while the visual bundle comes from
        # the multimodal query it replaced. The donor has the same relation
        # and is never the episode support instance.
        visual_item = item.get('_visual_donor', item)
        img_id = visual_item.get("img_id", "")
        h_name = visual_item["h"][0]
        t_name = visual_item["t"][0]

        has_visual = bool(img_id)
        if has_visual:
            img_ori = self._load_image(
                os.path.join(self.pic_path, f"{img_id}.jpg")
            )
        else:
            # Official FewRel text records do not have an image identifier.
            # Do not route them through PIL (which would emit one warning per
            # sampled item) and, more importantly, preserve that distinction
            # in the fifth reliability value below.
            img_ori = torch.zeros((3, 224, 224), dtype=torch.float32)

        if has_visual:
            obj_images, obj_weights = self._load_obj_images(
                img_id, img_ori, h_name, t_name
            )
        else:
            obj_images = torch.zeros((9, 224, 224), dtype=torch.float32)
            obj_weights = torch.zeros(5, dtype=torch.float32)

        caption_ids, caption_type_ids, caption_att_mask = self._tokenize_caption(
            img_id, h_name, t_name,
            fallback_token_ids=t[0].squeeze(0),
            fallback_att_mask=t[1].squeeze(0)
        )

        dummy_img     = torch.zeros((3, 224, 224), dtype=torch.float32)
        dummy_objects = torch.zeros((9, 224, 224), dtype=torch.float32)

        params = [
            t[0].squeeze(0),   # [0]  token
            t[1].squeeze(0),   # [1]  att_mask
            t[2].squeeze(0),   # [2]  pos1
            t[3].squeeze(0),   # [3]  pos2
            t[4].squeeze(0),   # [4]  token_phrase
            t[5].squeeze(0),   # [5]  att_mask_phrase
            dummy_img,         # [6]  image_diff         (3,224,224)  — 占位
            img_ori,           # [7]  image_ori          (3,224,224)  — 主图
            dummy_objects,     # [8]  image_diff_objects (9,224,224)  — 占位
            obj_images,        # [9]  image_ori_objects  (9,224,224)  ← 循环复用 crop
            obj_weights,       # [10] slot reliability + caption fraction + visual flag (5,)
            caption_ids,       # [11] caption_input_ids  ← 实体锚定 caption
            caption_type_ids,  # [12] caption_token_type_ids
            caption_att_mask,  # [13] caption_attention_mask
        ]

        packed_instance = [self.rel2id[rel_name], img_id] + params

        return packed_instance

    def _load_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            logging.warning(f"图像加载失败: {img_path}, 原因: {e}")
        return torch.zeros((3, 224, 224), dtype=torch.float32)

    def __len__(self):
        return self.episodes

    def __getitem__(self, index):
        current_N = min(self.N, len(self.active_classes))
        target_classes = random.sample(self.active_classes, current_N)

        support_set, query_set = [], []

        for class_name in target_classes:
            all_instances = self.all_json_data[class_name]
            num_available = len(all_instances)

            if num_available >= (self.K + self.Q):
                sampled = random.sample(all_instances, self.K + self.Q)
                support_instances = sampled[:self.K]
                query_instances   = sampled[self.K:]
            elif num_available >= self.K:
                support_instances = random.sample(all_instances, self.K)
                query_instances   = random.choices(all_instances, k=self.Q)
            else:
                support_instances = random.choices(all_instances, k=self.K)
                query_instances   = random.choices(all_instances, k=self.Q)

            supplementary_count = self.supplementary_queries_per_class
            if supplementary_count:
                visual_donors = query_instances[
                    self.Q - supplementary_count:
                ]
                supplementary_pool = self.supplementary_text_data[class_name]
                if len(supplementary_pool) >= supplementary_count:
                    supplementary_instances = random.sample(
                        supplementary_pool, supplementary_count
                    )
                else:
                    supplementary_instances = random.choices(
                        supplementary_pool, k=supplementary_count
                    )
                if self.supplementary_visual_mode == 'same_relation':
                    supplementary_instances = [
                        {**supplementary, '_visual_donor': donor}
                        for supplementary, donor in zip(
                            supplementary_instances, visual_donors
                        )
                    ]
                query_instances = query_instances[
                    :self.Q - supplementary_count
                ] + supplementary_instances
            for inst in support_instances:
                support_set.append(self._process_single_instance(inst, class_name))
            for inst in query_instances:
                query_set.append(self._process_single_instance(inst, class_name))

        return support_set, query_set

    @staticmethod
    def collate_fn(batch):
        support_set, query_set = batch[0]

        def pack(dataset_set):
            labels = torch.tensor([inst[0] for inst in dataset_set], dtype=torch.long)

            packed_args = []
            for i in range(2, 16):
                all_samples_field = [inst[i] for inst in dataset_set]
                stacked = torch.stack(all_samples_field, dim=0)
                if stacked.dim() == 3 and stacked.size(1) == 1:
                    stacked = stacked.squeeze(1)
                packed_args.append(stacked)

            if len(packed_args) != 14:
                raise ValueError(f"Collate 阶段失败：期望 14 个参数，实际 {len(packed_args)} 个")

            return labels, packed_args

        return pack(support_set), pack(query_set)


def FewRELFewShotLoader(
    data_paths, pic_path, rel2id, tokenizer,
    batch_size, shuffle, N, K, Q,
    is_train=True, num_workers=4,
    active_classes=None,
    caption_path=None,
    caption_mode='entity_scene',
    obj_path=None,
    obj_manifest_path=None,
    entity_obj_metadata_path=None,
    supplementary_text_path=None,
    supplementary_queries_per_class=0,
    supplementary_visual_mode='none',
    bert_tokenizer=None,
    episodes=None,
    **kwargs
):
    dataset = FewRELFewShotDataset(
        data_paths=data_paths,
        pic_path=pic_path,
        rel2id=rel2id,
        tokenizer=tokenizer,
        N=N, K=K, Q=Q,
        is_train=is_train,
        max_length=kwargs.get('max_length', 128),
        active_classes=active_classes,
        caption_path=caption_path,
        caption_mode=caption_mode,
        obj_path=obj_path,
        obj_manifest_path=obj_manifest_path,
        entity_obj_metadata_path=entity_obj_metadata_path,
        supplementary_text_path=supplementary_text_path,
        supplementary_queries_per_class=supplementary_queries_per_class,
        supplementary_visual_mode=supplementary_visual_mode,
        bert_tokenizer=bert_tokenizer,
        image_augmentation=kwargs.get('image_augmentation', 'none'),
        episodes=episodes,
        kwargs=kwargs
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=FewRELFewShotDataset.collate_fn
    )
