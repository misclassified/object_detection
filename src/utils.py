import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def convert_xml_to_csv(path):
    """Reads xml files with information on class name and bounding
    box and write them to a csv file

    Arguments:
        path = str, path where xml files are stored
    """

    # Inizialize class names vector and xml valies
    classes_names = []
    xml_values = []

    # Read in XML files and extract bounding box info from the DOM
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for member in root.findall("object"):
            classes_names.append(member[0].text)
            xml_values.append((
                root.find("filename").text,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                member[0].text,
                int(member.find('bndbox').find('xmin').text),
                int(member.find('bndbox').find('ymin').text),
                int(member.find('bndbox').find('xmax').text),
                int(member.find('bndbox').find('ymax').text),
            ))

    # Create a Pandas DataFrame
    column_name = ["filename", "width", "height", "class",
                   "xmin", "ymin", "xmax", "ymax",]
    xml_df = pd.DataFrame(xml_values, columns=column_name)
    classes_names = list(set(classes_names))
    classes_names.sort()
    
    return xml_df, classes_names
