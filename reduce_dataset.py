import os
import shutil

# Original dataset folder
SOURCE_DATASET = "dogs-vs-cats-classification"

# New reduced dataset folder
DESTINATION_DATASET = "small_dataset"

# Number of images to copy
LIMITS = {
    "train": 2000,
    "validation": 500,
    "test": 500
}

classes = ["cats", "dogs"]

for folder in LIMITS:
    for cls in classes:
        source_folder = os.path.join(SOURCE_DATASET, folder, cls)
        destination_folder = os.path.join(DESTINATION_DATASET, folder, cls)

        # Create destination folder
        os.makedirs(destination_folder, exist_ok=True)

        # Get image list
        images = os.listdir(source_folder)

        # Copy limited number of images
        for image in images[:LIMITS[folder]]:
            shutil.copy(
                os.path.join(source_folder, image),
                os.path.join(destination_folder, image)
            )

print("✅ Small dataset created successfully!")
print("Location:", DESTINATION_DATASET)