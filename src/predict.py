
from PIL import Image
import torch
import torch.nn.functional as F


def predict_image(
    image_path,
    model,
    transform,
    idx_to_class,
    device
):

    model.eval()

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(image)

    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = F.softmax(
            output,
            dim=1
        )

        confidence, predicted_index = (
            torch.max(
                probabilities,
                dim=1
            )
        )

    predicted_index = (
        predicted_index.item()
    )

    confidence = confidence.item()

    predicted_class = (
        idx_to_class[predicted_index]
    )

    return predicted_class, confidence
