class ImageFolder(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.train_images = list(map(lambda x: os.path.join(image_dir+'train', x), os.listdir(image_dir+'train')))
        self.test_images = list(map(lambda x: os.path.join(image_dir+'test', x), os.listdir(image_dir+'test')))
        self.transform = transform
        self.mode = mode

        test_path = 'test_eng.pkl'
        if os.path.isfile(test_path):
            self.test_dataset = torch.load(test_path)
        else:
            self.test_dataset = []
            self.preprocess()
            torch.save(self.test_dataset, test_path)

        if mode == 'train':
            self.num_images = len(self.train_images)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""

        random.seed(1234)
        random.shuffle(self.test_images)
        for i, img in enumerate(self.test_images):
            style_idx = int(img.split('_')[0][len(self.image_dir+'test/'):])
            char_idx = int(img.split('_')[1][:-len(".png")])

            target = random.choice([x for x in self.test_images
                                     if str(style_idx)+'_' in x and '_'+str(char_idx) not in x])
            char_trg_idx = int(target.split('_')[1][:-len(".png")])

            self.test_dataset.append([img, style_idx, char_idx, target, char_trg_idx])

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if self.mode == 'train':
            random.seed()
            random.shuffle(self.train_images)

            src = self.train_images[index]
            src_style = int(src.split('_')[0][len(self.image_dir+'train/'):])
            src_char = int(src.split('_')[1][:-len(".png")])

            try:
                trg = random.choice([x for x in self.train_images
                                        if str(src_style)+'_' in x and '_'+str(src_char) not in x])
            except:
                trg = src
            trg_char = int(trg.split('_')[1][:-len(".png")])
        else:
            src, src_style, src_char, trg, trg_char = self.test_dataset[index]

        src = self.transform(Image.open(src))
        trg = self.transform(Image.open(trg))

        return src, src_style, src_char, \
               trg, trg_char

    def __len__(self):
        """Return the number of images."""
return self.num_images