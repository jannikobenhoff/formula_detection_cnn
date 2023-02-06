import torch
import torchvision
from torchvision import transforms
from PIL import Image
from Neural.data_loader import shutil_or_just_labels

labels = {'!': 0, '(': 1, ')': 2, '+': 3, ',': 4, '-': 5, '0': 6, '1': 7, '2': 8, '3': 9,
          '4': 10, '5': 11, '6': 12, '7': 13, '8': 14, '9': 15, '=': 16, 'a': 17, 'c': 18,
          'g': 19, 'H': 20, 'M': 21, 'n': 22, 'R': 23, 'S': 24, 'T': 25, '[': 26, ']': 27,
          'b': 28, 'd': 29, '/': 30, 'e': 31, 'f': 32, 'i': 33, 'k': 34, 'l': 35, 'o': 36,
          'p': 37, 'q': 38, '*': 39, 'u': 40, 'v': 41, 'w': 42, 'y': 43, '{': 44, '}': 45}
def predict(img_list):
    model = torch.jit.load("/Users/jannikobenhoff/Documents/pythonProjects/quantum_computation/testing/model_stroke.torch")
    predictions = []
    for img in img_list:
        img = img.imagearray
        custom_image = torch.from_numpy(img).type(torch.float32) / 255
        custom_image = custom_image.unsqueeze(0)

        model.eval()
        with torch.inference_mode():
            custom_pred = model(custom_image.unsqueeze(dim=0))

        custom_image_pred_probs = torch.softmax(custom_pred, dim=1)
        print(custom_image_pred_probs)

        custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
        custom_image_pred_label = list(labels.keys())[custom_image_pred_label.data]
        predictions.append(custom_image_pred_label)
        #print("Custom Image Prediction: " + str(custom_image_pred_label))
    print(predictions)
    return "".join(predictions)

    # if show_image:
    #     plt.imshow(custom_image.squeeze().permute(1, 2, 0))
    #     plt.title("Prediction: " + custom_image_pred_label + " True: " + symbol)
    #     plt.axis = False
