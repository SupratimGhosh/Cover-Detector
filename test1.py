from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="tTodufarW9yFlMaSiAuA"
)

result = client.run_workflow(
    workspace_name="trial-kgruu",
    workflow_id="custom-workflow",
    images={
        "image": "test.png"
    },
    use_cache=True # cache workflow definition for 15 minutes
)

print(type(result))
print(result)

# If result is a list, get the first item
if isinstance(result, list) and len(result) > 0:
    output = result[0]
else:
    output = result




def annotate_and_show_image(image_path, result):
    import base64
    import io
    from PIL import Image, ImageDraw

    # Open the original image
    image = Image.open(image_path).convert("RGB")

    # Extract bounding box info
    preds = result[0]["output"]["predictions"]
    draw = ImageDraw.Draw(image)
    for pred in preds:
        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]
        # Convert center x,y and width,height to box corners
        left = x - w/2
        top = y - h/2
        right = x + w/2
        bottom = y + h/2
        draw.rectangle([left, top, right, bottom], outline="red", width=4)
        # Optionally, draw class label
        if "class" in pred:
            draw.text((left, top), pred["class"], fill="red")
    image.show()

# Usage:
#annotate_and_show_image("test.png", result)


def crop_and_show_annotated_region(image_path, result, size=(320, 320)):
    from PIL import Image
    # Open the original image
    image = Image.open(image_path).convert("RGB")
    preds = result[0]["output"]["predictions"]
    if not preds:
        print("No predictions found.")
        return None
    # Use the first prediction
    pred = preds[0]
    x = pred["x"]
    y = pred["y"]
    w = pred["width"]
    h = pred["height"]
    left = int(x - w/2)
    top = int(y - h/2)
    right = int(x + w/2)
    bottom = int(y + h/2)
    # Crop and resize
    cropped = image.crop((left, top, right, bottom))
    cropped = cropped.resize(size)
    cropped.show()
    return cropped

# Usage:
cropped_img = crop_and_show_annotated_region("test.png", result)


def show_image_and_print_click_coords(image_or_path):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    if isinstance(image_or_path, str):
        image = Image.open(image_or_path).convert("RGB")
    else:
        image = image_or_path
    arr = np.array(image)

    fig, ax = plt.subplots()
    ax.imshow(arr)
    ax.set_title("Click on the image to print coordinates. Close window to finish.")

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            print(f"Clicked at: x={int(event.xdata)}, y={int(event.ydata)}")

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

# Usage:
# if cropped_img is not None:
    # show_image_and_print_click_coords(cropped_img)



def crop_image_with_polygon(image, polygon_coords, size=(320, 320)):
    from PIL import Image, ImageDraw
    import numpy as np
    # Create a mask for the polygon
    mask = Image.new('L', image.size, 0)
    ImageDraw.Draw(mask).polygon(polygon_coords, outline=1, fill=255)
    # Apply mask to image
    result = Image.new('RGB', image.size)
    result.paste(image, mask=mask)
    # Crop to bounding box of polygon
    xs, ys = zip(*polygon_coords)
    bbox = (min(xs), min(ys), max(xs), max(ys))
    cropped = result.crop(bbox)
    # Stretch to square of given size
    cropped = cropped.resize(size)
    cropped.show()
    return cropped

#Example usage:
polygon = [(68, 41), (9, 104), (300, 99), (262, 36)]
if cropped_img is not None:
    final_crop=crop_image_with_polygon(cropped_img, polygon)


#testing (not for main code)
#show_image_and_print_click_coords(final_crop)


def analyze_material_color_and_texture(image, polygon_coords, debug=True):
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt

    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    if img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
    if debug:
        print(f"Original image shape: {img_cv.shape}")

    # 1. Preprocessing: CLAHE (adaptive histogram equalization) for contrast, gamma correction for brightness
    lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    if debug:
        plt.figure(); plt.title('CLAHE'); plt.imshow(img_clahe); plt.show()

    # Gamma correction (reduce glare, normalize brightness)
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(img_clahe, table)
    if debug:
        plt.figure(); plt.title('Gamma Corrected'); plt.imshow(img_gamma); plt.show()

    # 2. Create polygon mask
    mask = np.zeros(img_gamma.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon_coords, dtype=np.int32)], 255)
    if debug:
        plt.figure(); plt.title('Polygon Mask'); plt.imshow(mask, cmap='gray'); plt.show()

    # 3. Count black/gray and color pixels within the polygon
    masked_img = cv2.bitwise_and(img_gamma, img_gamma, mask=mask)
    if debug:
        plt.figure(); plt.title('Masked Image'); plt.imshow(masked_img); plt.show()

    # Define thresholds for "nearly black" and "color"
    # Black/gray: all channels < 60, or stddev of channels < 15 (grayish)
    black_thresh = 60
    gray_stddev_thresh = 15
    img_flat = masked_img[mask == 255]
    if debug:
        print(f"Total polygon pixels: {img_flat.shape[0]}")
    black_pixels = 0
    color_pixels = 0
    for px in img_flat:
        if np.all(px < black_thresh):
            black_pixels += 1
        elif np.std(px) < gray_stddev_thresh and np.mean(px) < 100:
            black_pixels += 1
        else:
            color_pixels += 1
    total = black_pixels + color_pixels
    percent_black = (black_pixels / total * 100) if total > 0 else 0
    percent_color = (color_pixels / total * 100) if total > 0 else 0
    print(f"Black/gray pixels: {black_pixels}")
    print(f"Color pixels: {color_pixels}")
    print(f"Percent black/gray: {percent_black:.2f}%")
    print(f"Percent color: {percent_color:.2f}%")
    return percent_black, percent_color

# Example usage:
if final_crop is not None:
    analyze_material_color_and_texture(final_crop, [(70,32), (275,19), (316,291), (3,317)])

