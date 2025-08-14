import os
import xml.etree.ElementTree as ET

# Directory paths
annotations_dir = 'dataset/annotations'
images_dir = 'dataset/images'
labels_dir = 'dataset/labels'

# Define class names
classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']

# Create labels directory if it doesn't exist
os.makedirs(labels_dir, exist_ok=True)

# Process all XML annotation files
for filename in os.listdir(annotations_dir):
    if not filename.endswith('.xml'):
        continue

    xml_path = os.path.join(annotations_dir, filename)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image size
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # Convert each object to YOLO format
    txt_lines = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)

        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        # Convert to YOLO format (normalized)
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        txt_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Write to .txt file with the same name as the XML file
    txt_filename = filename.replace('.xml', '.txt')
    txt_path = os.path.join(labels_dir, txt_filename)
    with open(txt_path, 'w') as f:
        f.write('\n'.join(txt_lines))
