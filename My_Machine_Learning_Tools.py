# Machine_Learning_Tools.py
#by Till Heinrich

import pandas as pd
from itertools import chain
import torch
import os
import pickle

class LabelEncoder:
    """
    Eine Klasse, die verwendet wird, um eine Zuordnung von Labels zu numerischen Indizes und umgekehrt zu erstellen.
    Dies ist nützlich für maschinelles Lernen und andere Anwendungen, bei denen kategorische Daten in eine numerische Form umgewandelt werden müssen.
    """

    def __init__(self):
        """
        Initialisiert ein neues LabelEncoder-Objekt mit leeren Zuordnungen und einer Anzahl von Klassen (num_classes) von Null.
        """
        self.label_to_index = {}
        self.index_to_label = {}
        self.num_classes = 0
    
    def fit(self, labels):
        """
        Passt den Encoder an eine Liste von Labels an, indem er eindeutige Labels identifiziert und ihnen Indizes zuweist.
        
        Parameters:
        labels (list of str): Die Liste der Labels, die angepasst werden sollen.
        """
        unique_labels = sorted(set(labels))
        self.num_classes = len(unique_labels)
        
        for index, label in enumerate(unique_labels):
            self.label_to_index[label] = index
            self.index_to_label[index] = label
    
    def transform(self, labels):
        """
        Transformiert eine Liste von Labels in eine Liste von Indizes basierend auf den während des Fits gelernten Zuordnungen.
        
        Parameters:
        labels (list of str): Die Liste der Labels, die transformiert werden sollen.
        
        Returns:
        list of int: Die Liste der entsprechenden Indizes.
        """
        return [self.label_to_index[label] for label in labels]
    
    def inverse_transform(self, indices):
        """
        Transformiert eine Liste von Indizes zurück in die ursprünglichen Labels.
        
        Parameters:
        indices (list of int): Die Liste der Indizes, die zurücktransformiert werden sollen.
        
        Returns:
        list of str: Die Liste der entsprechenden Labels.
        """
        return [self.index_to_label[index] for index in indices]
    
    def transform_fit(self, labels):
        """
        Passt den Encoder an die Labels an und transformiert sie anschließend in Indizes.
        
        Parameters:
        labels (list of str): Die Liste der Labels, die angepasst und transformiert werden sollen.
        
        Returns:
        list of int: Die Liste der entsprechenden Indizes.
        """
        self.fit(labels)
        return self.transform(labels)
    
    def extend(self, new_labels):
        """
        Erweitert die Zuordnung um neue Labels, die in den bisherigen Anpassungen nicht enthalten waren.
        
        Parameters:
        new_labels (list of str): Die Liste der neuen Labels, die zur Zuordnung hinzugefügt werden sollen.
        """
        existing_labels = set(self.label_to_index.keys())
        unique_new_labels = sorted(set(new_labels))
        for label in unique_new_labels:
            if label not in existing_labels:
                self.label_to_index[label] = len(self.label_to_index)
                self.index_to_label[len(self.index_to_label)] = label
                existing_labels.add(label)
        self.num_classes = len(self.label_to_index)
    
    def extend_transform(self, new_labels):
        """
        Erweitert die Zuordnung um neue Labels und transformiert diese anschließend in Indizes.
        
        Parameters:
        new_labels (list of str): Die Liste der neuen Labels, die erweitert und transformiert werden sollen.
        
        Returns:
        list of int: Die Liste der entsprechenden Indizes.
        """
        self.extend(new_labels)
        return self.transform(new_labels)
    
    def load(self, filepath):
        """
        Lädt einen gespeicherten Encoder von einer Datei.
        
        Parameters:
        filepath (str): Der Pfad zur Datei, aus der der Encoder geladen werden soll.
        """
        with open(filepath, 'rb') as f:
            encoder = pickle.load(f)
        self.label_to_index = encoder.label_to_index
        self.index_to_label = encoder.index_to_label
        self.num_classes = encoder.num_classes
    
    def save(self, filepath):
        """
        Speichert den aktuellen Encoder in einer Datei.
        
        Parameters:
        filepath (str): Der Pfad zur Datei, in die der Encoder gespeichert werden soll.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    def __len__(self):
        """
        Gibt die Anzahl der einzigartigen Klassen (Labels) zurück.
        
        Returns:
        int: Die Anzahl der einzigartigen Klassen.
        """
        return self.num_classes
    
    def labels(self):
        """
        Gibt die aktuelle Zuordnung von Labels zu Indizes und umgekehrt zurück.
        
        Returns:
        dict: Ein Wörterbuch mit zwei Schlüsseln, `label_to_index` und `index_to_label`.
        """
        return {
            'label_to_index': self.label_to_index,
            'index_to_label': self.index_to_label
        }

def df_unique_list_values(df, column):
    """
    Sammelt alle eindeutigen Werte aus der angegebenen Spalte eines DataFrames.

    Parameters:
    df (pd.DataFrame): Der DataFrame, aus dem die Werte gesammelt werden sollen.
    column (str): Der Name der Spalte, die analysiert werden soll.

    Returns:
    list: Eine Liste eindeutiger Werte aus der angegebenen Spalte.
    """
    unique_values = set()
    # Extrahiere Werte aus der Spalte und füge sie der Menge hinzu
    unique_values.update(chain.from_iterable(df[column].apply(lambda x: x if isinstance(x, list) else [x])))
    return list(unique_values)

def df_explode_listcolumn(df, column, possible_values=None):
    """
    Explodiert eine Listen-Spalte eines DataFrames und erstellt Spalten für jeden möglichen Wert. Alternativ kann eine Liste der möglichen Werte angegeben werden.

    Parameters:
    df (pd.DataFrame): Der DataFrame, der die zu explodierende Spalte enthält.
    column (str): Der Name der zu explodierenden Spalte.
    possible_values (list, optional): Wird eine Liste der möglichen Werte angegeben, so werden nur diese explodiert. Defaults to None.

    Returns:
    pd.DataFrame: Der DataFrame mit den neuen, explodierten Spalten.
    """

    if possible_values is None:
        # Ursprüngliche Funktionsweise
        df = df.join(pd.crosstab((s := df[column].explode()).index, s).add_prefix(column + '_'))
        df = df.drop(columns=column)
        for col in [col for col in df if col.startswith(column + '_')]:
            df[col].fillna(value=0, inplace=True)
    else:
        # Erstellt leere Spalten für alle möglichen Werte und füllt sie mit 0
        columns_to_add = [f"{column}_{value}" for value in possible_values]
        df = pd.concat([df, pd.DataFrame(columns=columns_to_add)], axis=1)
        df[columns_to_add] = 0
        
        # Füllt die Spalten mit 1, falls der Wert in der Liste vorhanden ist
        for idx, row in df.iterrows():
            for value in row[column]:
                if value in possible_values:
                    df.at[idx, f"{column}_{value}"] = 1
        
        # Löscht die ursprüngliche Spalte
        df = df.drop(columns=column)
    
    return df

def df_string_to_list(df, column, entry_delimiter='\'', separator=',', list_start='[', list_end=']'):
    """
    Konvertiert eine Pandas-Spalte, die Listen als Strings formatiert enthält, in echte Listen.

    Parameters:
    df (pd.DataFrame): Der DataFrame, der die zu konvertierende Spalte enthält.
    column (str): Der Name der zu konvertierenden Spalte.
    entry_delimiter (str): Das Zeichen, das die Einträge innerhalb der Listen umgibt (Standard: '\'').
    separator (str): Das Zeichen, das die Einträge innerhalb der Listen abtrennt (Standard: ' ').
    list_start (str): Das Zeichen, das den Anfang der Liste markiert (Standard: '[').
    list_end (str): Das Zeichen, das das Ende der Liste markiert (Standard: ']').

    Returns:
    pd.DataFrame: Der DataFrame mit der konvertierten Spalte.
    """
    def convert_to_list(cell):
        if isinstance(cell, str):
            # Entferne die Anfangs- und Endzeichen der Liste
            cell = cell.strip(list_start + list_end)
            # Splitte die Einträge der Liste anhand des separators und entferne das entry_delimiter
            cell = [entry.strip(entry_delimiter) for entry in cell.split(entry_delimiter+separator+entry_delimiter) if entry.strip(entry_delimiter)]
        return cell

    # Wende die Konvertierung auf die Spalte an
    df[column] = df[column].apply(convert_to_list)
    
    return df


def _modelversions_get_available_versions(model_path_base):
    model_dir = os.path.dirname(model_path_base)
    model_base_name = os.path.basename(model_path_base)
    
    # Liste alle Dateien im Verzeichnis auf und filtere diejenigen, die mit dem Basisnamen beginnen und das erwartete Muster haben
    versions = [int(f.split('_v')[-1].split('.pt')[0]) for f in os.listdir(model_dir) 
                if f.startswith(model_base_name) and '_v' in f and f.split('_v')[-1].split('.pt')[0].isdigit()]
    
    versions.sort()
    return versions


def _modelversions_format_versions(versions):
    """
    Formatiert eine Liste von Versionsnummern zu einem lesbaren String von zusammenhängenden Bereichen.

    Parameters:
    versions (list of int): Eine sortierte Liste von Versionsnummern.

    Returns:
    str: Ein formatierter String, der die zusammenhängenden Bereiche von Versionsnummern darstellt.

    Example:
    >>> format_versions([1, 2, 3, 5, 6, 7, 9])
    '1-3, 5-7, 9'
    
    >>> format_versions([1, 2, 3])
    '1-3'

    >>> format_versions([1])
    '1'
    """
    formatted_versions = []
    start = versions[0]
    end = versions[0]

    for i in range(1, len(versions)):
        if versions[i] == end + 1:
            end = versions[i]
        else:
            if start == end:
                formatted_versions.append(str(start))
            else:
                formatted_versions.append(f"{start}-{end}")
            start = versions[i]
            end = versions[i]

    # Handle the last range
    if start == end:
        formatted_versions.append(str(start))
    else:
        formatted_versions.append(f"{start}-{end}")

    return ", ".join(formatted_versions)

def _modelversions_get_model_path_with_version(model_path_base, version):
    """
    Erzeugt den Dateipfad für eine spezifische Version eines Modells.

    Parameters:
    model_path_base (str): Der Basispfad des Modells ohne Versionsnummer und Dateiendung.
    version (int): Die Versionsnummer des Modells.

    Returns:
    str: Der vollständige Dateipfad zur angegebenen Modellversion, einschließlich der Dateierweiterung '.pt'.

    Example:
    >>> _modelversions_get_model_path_with_version('/path/to/models/test', 3)
    '/path/to/models/test_v3.pt'

    >>> _modelversions_get_model_path_with_version('/path/to/models/test', 1)
    '/path/to/models/test_v1.pt'
    """
    return f"{model_path_base}_v{version}.pt"

def modelversions_save_model(model, model_path_base):
    """
    Speichert ein Modell und inkrementiert automatisch die Versionsnummer.

    Parameters:
    model (torch.nn.Module): Das PyTorch-Modell, das gespeichert werden soll.
    model_path_base (str): Der Basispfad, unter dem das Modell gespeichert werden soll. Die Versionsnummer wird automatisch angehängt.

    Returns:
    None

    Example:
    >>> model = MyModel()
    >>> modelversions_save_model(model, '/path/to/models/test')
    Model saved as /path/to/models/test_v1.pt
    """
    available_versions = _modelversions_get_available_versions(model_path_base)
    new_version = (available_versions[-1] + 1) if available_versions else 1
    model_path = _modelversions_get_model_path_with_version(model_path_base, new_version)
    torch.save({'model_state_dict': model.state_dict()}, model_path)
    print(f"Model saved as {model_path}")


def modelversions_load_model(model, model_path_base):
    """
    Lädt ein Modell aus einer spezifischen Version.

    Parameters:
    model (torch.nn.Module): Das PyTorch-Modell, das geladen werden soll.
    model_path_base (str): Der Basispfad, unter dem das Modell gespeichert ist.

    Returns:
    None

    Example:
    >>> model = MyModel()
    >>> load_model(model, '/path/to/models/test')
    Available versions: [1, 2, 3]
    Enter 0 to cancel.
    Enter the version to load (1-3 or 0 to cancel): 2
    Model loaded from /path/to/models/test_v2.pt
    """
    available_versions = _modelversions_get_available_versions(model_path_base)
    if available_versions!=[]:
        while True:
            try:
                version = int(input(f"Enter the version to load ({_modelversions_format_versions(available_versions)} or 0 to cancel): "))
                if version == 0:
                    print("Loading cancelled.")
                    return
                elif version in available_versions:
                    break
                else:
                    print(f"Invalid version. Please enter a number out of {_modelversions_format_versions(available_versions)}, or 0 to cancel.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
    else:
        print("No versions available.")
        return
    
    model_path = _modelversions_get_model_path_with_version(model_path_base, version)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {model_path}")