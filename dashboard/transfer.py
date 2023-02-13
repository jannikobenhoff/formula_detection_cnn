import torch
import torchvision
from torchvision import transforms
from PIL import Image
from Neural.data_loader import shutil_or_just_labels

labels = {'!': 0, '(': 1, ')': 2, '+': 3, ',': 4, '-': 5, '0': 6, '1': 7, '2': 8,
          '3': 9, '4': 10, '5': 11, '6': 12, '7': 13, '8': 14, '9': 15, '=': 16,
          'A': 17, 'C': 18, 'Delta': 19, 'G': 20, 'H': 21, 'M': 22, 'N': 23, 'R': 24,
          'S': 25, 'T': 26, 'X': 27, '[': 28, ']': 29, 'alpha': 30, 'ascii_124': 31,
          'b': 32, 'beta': 33, 'cos': 34, 'd': 35, '/': 36, 'e': 37, 'exists': 38,
          'f': 39, 'forall': 40, 'forward_slash': 41, 'gamma': 42, 'geq': 43, 'gt': 44,
          'i': 45, 'in': 46, 'infty': 47, 'int': 48, 'j': 49, 'k': 50, 'l': 51,
          'lambda': 52, 'ldots': 53, 'leq': 54, 'lim': 55, 'log': 56, 'lt': 57,
          'mu': 58, 'neq': 59, 'o': 60, 'p': 61, 'phi': 62, 'pi': 63, 'pm': 64,
          'prime': 65, 'q': 66, 'rightarrow': 67, 'sigma': 68, 'sin': 69, 'sqrt': 70,
          'sum': 71, 'tan': 72, 'theta': 73, 'times': 74, 'u': 75, 'v': 76, 'w': 77,
          'y': 78, 'z': 79, '{': 80, '}': 81}

def predict(img_list):
    model = torch.jit.load("/Users/jannikobenhoff/Documents/pythonProjects/quantum_computation/testing/model_thick.torch")
    predictions = []
    for img in img_list:
        img = img.imagearray
        custom_image = torch.from_numpy(img).type(torch.float32) / 255
        custom_image = custom_image.unsqueeze(0)

        model.eval()
        with torch.inference_mode():
            custom_pred = model(custom_image.unsqueeze(dim=0))

        custom_image_pred_probs = torch.softmax(custom_pred, dim=1)

        custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
        custom_image_pred_label = list(labels.keys())[custom_image_pred_label.data]
        predictions.append(custom_image_pred_label)
        #print("Custom Image Prediction: " + str(custom_image_pred_label))
    print("Prediction: ", predictions)
    return "".join(predictions)

    # if show_image:
    #     plt.imshow(custom_image.squeeze().permute(1, 2, 0))
    #     plt.title("Prediction: " + custom_image_pred_label + " True: " + symbol)
    #     plt.axis = False
