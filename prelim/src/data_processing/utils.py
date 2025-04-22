from PIL import Image

def get_images(images, mean, std, aug=True, cols=8):
    """
    Return images from the tensor provided.
    Inverse Normalize the images if aug is True
    """
    import torchvision.transforms as transforms
    trnsfrms = []
    if aug:
        trnsfrms.append(transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]))
    import torchvision.transforms as transforms
    
    trnsfrms.append(transforms.ToPILImage())
    tr = transforms.Compose(trnsfrms)
    ims1 = []
    for img_idx in range(len(images)):
        pil_img1 = tr(images[img_idx])
        ims1.append(pil_img1)
    
    def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
        """hconcat images"""
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    
    def get_concat_h_multi_blank(im_list):
        """sdfs"""
        _im = im_list.pop(0)
        for im in im_list:
            _im = get_concat_h_blank(_im, im)
        return _im
    
    def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
        """hconcat images"""
        dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst
    
    def get_concat_v_multi_blank(im_list):
        """sdfs"""
        _im = im_list.pop(0)
        for im in im_list:
            _im = get_concat_v_blank(_im, im)
        return _im
    
    hor_images = []
    for i in range(len(ims1) // cols):
        if len(ims1[i * cols:min(i * cols + cols, len(ims1))]) == cols:
            hor_images.append(get_concat_h_multi_blank(ims1[i * cols:min(i * cols + cols, len(ims1))]))
    
    return get_concat_v_multi_blank(hor_images)