import pandas as pd
import os

# Path to the remaining (non-seed) metadata CSV
_CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "MetaData of Dataset (not seed data).csv"
)

# Load once when the module is imported
_metadata = pd.read_csv(_CSV_PATH)


def get_diagnosis(image_name: str) -> str:
    """
    Simulated oracle: looks up the diagnosis (dx) for a given image.

    Parameters
    ----------
    image_name : str
        The image identifier, e.g. 'ISIC_0027419' or 'ISIC_0027419.jpg'.

    Returns
    -------
    str
        The diagnosis code (e.g. 'mel', 'bcc', 'nv', 'bkl', 'akiec', 'df', 'vasc').

    Raises
    ------
    ValueError
        If the image is not found in the remaining metadata.
    """
    # Strip file extension if provided
    image_id = image_name.replace(".jpg", "").replace(".png", "")

    match = _metadata[_metadata["image_id"] == image_id]

    if match.empty:
        raise ValueError(
            f"Image '{image_id}' not found in the remaining metadata. "
            "It may already be in the seed set or does not exist."
        )

    return match.iloc[0]["dx"]


# Quick self-test when run directly
if __name__ == "__main__":
    test_id = _metadata.iloc[0]["image_id"]
    result = get_diagnosis(test_id)
    print(f"Oracle lookup  ->  image: {test_id}  |  diagnosis: {result}")
