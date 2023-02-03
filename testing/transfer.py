import torch
import torchvision
from torchvision import transforms
from PIL import Image

def predict(img_list):
    model = torch.jit.load("model.torch")
    img = img_list[0].imagearray
    img = Image.fromarray(img) # .convert("RGB")
    img.save("custom.jpg")
    print(len(img_list))
    # tensor = torch.from_numpy(img)
    custom_image = torchvision.io.read_image("custom.jpg").type(torch.float32) / 255
    custom_transforms = transforms.Compose([transforms.Resize((28, 28))])
    custom_image = custom_transforms(custom_image)

    model.eval()
    with torch.inference_mode():
        custom_pred = model(custom_image.unsqueeze(dim=0))

    custom_image_pred_probs = torch.softmax(custom_pred, dim=1)
    custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
    #custom_image_pred_label = labels[custom_image_pred_label]

    print("Custom Image Prediction: " + str(custom_image_pred_label))
    #print("True Symbol: ", symbol)

    # if show_image:
    #     plt.imshow(custom_image.squeeze().permute(1, 2, 0))
    #     plt.title("Prediction: " + custom_image_pred_label + " True: " + symbol)
    #     plt.axis = False
