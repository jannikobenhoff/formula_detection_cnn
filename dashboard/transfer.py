import torch
import torchvision
from torchvision import transforms
from PIL import Image
from Neural.data_loader import shutil_or_just_labels

labels = {0: '!', 1: '(', 2: ')', 3: '+', 5: ',', 6: '-', 7: '0', 8: '1', 9: '2', 10: '3', 11: '4', 12: '5',
          13: '6', 14: '7', 15: '8', 16: '9', 17: '=', 18: 'a', 19: 'b', 20: 'd', 21: 'div', 22: 'e', 23: 'f',
          24: 'i', 25: 'l', 26: 'n', 27: 'o', 28: 'p', 29: '*'}
def predict(img_list):
    model = torch.jit.load("/Users/jannikobenhoff/Documents/pythonProjects/quantum_computation/testing/model_stroke.torch")
    predictions = []
    for img in img_list:
        img = img.imagearray
        custom_image = torch.from_numpy(img).type(torch.float32) / 255
        custom_image = custom_image.unsqueeze(0)

        # img = Image.fromarray(img)  # .convert("L") # .convert("RGB")
        # img.save("custom.jpg")
        #
        # custom_image = torchvision.io.read_image("custom.jpg").type(torch.float32) / 255
        # torch.reshape(custom_image, (1, 28, 28))
        # print(custom_image.shape)
        # print(type(custom_image))
        # custom_image = torchvision.io.read_image("/Users/jannikobenhoff/Documents/pythonProjects/quantum_computation/testing/s.jpg").type(torch.float32) / 255
        # custom_transforms = transforms.Compose([transforms.Resize((28, 28))])
        # custom_image = custom_transforms(custom_image)

        model.eval()
        with torch.inference_mode():
            custom_pred = model(custom_image.unsqueeze(dim=0))

        custom_image_pred_probs = torch.softmax(custom_pred, dim=1)
        custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
        custom_image_pred_label = list(labels.values())[custom_image_pred_label.data]
        predictions.append(custom_image_pred_label)
        #print("Custom Image Prediction: " + str(custom_image_pred_label))
    print(predictions)
    return "".join(predictions)

    # if show_image:
    #     plt.imshow(custom_image.squeeze().permute(1, 2, 0))
    #     plt.title("Prediction: " + custom_image_pred_label + " True: " + symbol)
    #     plt.axis = False
