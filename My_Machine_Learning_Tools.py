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
    Funktionen zum speichern, laden und erweitern ermöglichen konsistente Indizes über verschiedene Datensätze.
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
        indices (list of int) oder torch.Tensor: Die Liste der Indizes oder Tensoren, die zurücktransformiert werden sollen.
        
        Returns:
        list of str oder torch.Tensor: Die Liste der entsprechenden Labels oder Tensoren.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        
        labels = [self.index_to_label[index] for index in indices]
        
        return labels
    
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

class ZScoreEncoder:
    def __init__(self):
        self.params = {}

    def fit(self, df, columns):
        """
        Berechnet den Mittelwert und die Standardabweichung der angegebenen Spalten.

        Parameters:
        df (pd.DataFrame): Der DataFrame, der die Daten enthält.
        columns (list): Liste der Spaltennamen, die standardisiert werden sollen.

        Returns:
        None
        """
        for column in columns:
            mean = df[column].mean()
            std = df[column].std()
            self.params[column] = {'mean': mean, 'std': std}

    def transform(self, df):
        """
        Standardisiert die zuvor in `fit` angegebenen Spalten basierend auf den gespeicherten Mittelwerten und Standardabweichungen.

        Parameters:
        df (pd.DataFrame): Der DataFrame, der die Daten enthält.

        Returns:
        pd.DataFrame: Der DataFrame mit den standardisierten Spalten.
        """
        if not self.params:
            raise ValueError("The encoder has not been fitted yet.")
        for column, stats in self.params.items():
            if column not in df.columns:
                raise ValueError(f"The column '{column}' is not present in the DataFrame.")
            mean = stats['mean']
            std = stats['std']
            df[column] = (df[column] - mean) / std
        return df

    def fit_transform(self, df, columns):
        """
        Berechnet den Mittelwert und die Standardabweichung und standardisiert die Spalten.

        Parameters:
        df (pd.DataFrame): Der DataFrame, der die Daten enthält.
        columns (list): Liste der Spaltennamen, die standardisiert werden sollen.

        Returns:
        pd.DataFrame: Der DataFrame mit den standardisierten Spalten.
        """
        self.fit(df, columns)
        return self.transform(df)

    def save(self, file_path):
        """
        Speichert die Mittelwert- und Standardabweichungsparameter in einer Datei.

        Parameters:
        file_path (str): Der Pfad zur Datei, in der die Parameter gespeichert werden sollen.

        Returns:
        None
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.params, f)

    def load(self, file_path):
        """
        Lädt die Mittelwert- und Standardabweichungsparameter aus einer Datei.

        Parameters:
        file_path (str): Der Pfad zur Datei, aus der die Parameter geladen werden sollen.

        Returns:
        None
        """
        with open(file_path, 'rb') as f:
            self.params = pickle.load(f)

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

def _modelversions_validate_save_params(mode,model_path_base,path):
    """
    Validiert die Eingabeparameter für das Speichern eines Modells basierend auf dem angegebenen Modus.

    Diese Funktion überprüft, ob die notwendigen Parameter für das Speichern eines Modells vorhanden und gültig sind, 
    abhängig vom Modus, in dem das Modell gespeichert werden soll.
    """
    if mode in ['path'] and path is None:
        print('Please enter a valid path')
        return False
    if mode in ['versions'] and model_path_base is None:
        print('Please enter a valid model_path_base')
        return False
    return True

def modelversions_save_model(model,optimizer, mode='versions', model_path_base=None, path=None, init_params=None):
    """
    Speichert ein Modell und seinen Optimiererzustand entweder an einem spezifischen Pfad oder mit einer automatisch inkrementierten Versionsnummer.
    Es ist möglich das Modell auf Basis der Datei zu laden ohne Informationen über die Klasse und die Initialisierungsparameter zu haben.

    Diese Funktion speichert das Modell und seinen Optimiererzustand zusammen mit den Modell- und Optimierer-Klasseninformationen 
    und den Initialisierungsparametern. Je nach Modus wird das Modell entweder an einem angegebenen Pfad gespeichert oder die 
    Versionsnummer wird automatisch erhöht und das Modell entsprechend gespeichert.

    Parameters:
    model (torch.nn.Module): Das PyTorch-Modell, das gespeichert werden soll. 
                             Das Modell sollte eine `state_dict`-Methode besitzen und seine Klasse sollte über eine
                             `__class__`-Eigenschaft verfügen.
                             
    optimizer (torch.optim.Optimizer): Der Optimierer, dessen Zustand zusammen mit dem Modell gespeichert werden soll.

    mode (str, optional): Der Modus, in dem das Modell gespeichert werden soll. Bestimmt, wie der Speicherpfad generiert wird.
                          Mögliche Werte sind:
                          - 'path': Ein vollständiger Pfad zum Modell wird verwendet.
                          - 'versions': Die Versionsnummer des Modells wird automatisch inkrementiert.
                          Standardwert ist 'versions'.

    model_path_base (str, optional): Der Basis-Pfad, unter dem das Modell gespeichert werden soll. 
                                     Erforderlich für den Modus 'versions'.

    path (str, optional): Der vollständige Pfad zum Modell. 
                          Erforderlich, wenn der Modus 'path' gewählt wird.

    init_params (dict, optional): Initialisierungsparameter des Modells, die beim Erstellen des Modells verwendet wurden. 
                                   Falls diese nicht angeben werden, wird angenommen, dass das Modell eine `init_params`-Eigenschaft hat.

    Returns:
    None: Gibt `None` zurück, wenn der Speicherprozess aufgrund ungültiger Parameter abgebrochen wurde.

    Example:
    >>> model = MultilingualBERTClassifier()
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> modelversions_save_model(model, optimizer, 'versions', model_path_base='/path/to/models')
    Model saved as /path/to/models_model_v1.pt
    
    >>> modelversions_save_model(model, optimizer, 'path', path='/path/to/models/model_v1.pt')
    Model saved as /path/to/models/model_v1.pt
    """
    if mode=='path':
        if not _modelversions_validate_save_params(mode,model_path_base,path):
            return None
        model_path=path
    elif mode=='versions':
        if not _modelversions_validate_save_params(mode,model_path_base,path):
            return None
        available_versions = _modelversions_get_available_versions(model_path_base)
        new_version = (available_versions[-1] + 1) if available_versions else 1
        model_path = _modelversions_get_model_path_with_version(model_path_base, new_version)
    else:
        print('Please enter a valid mode')
        return None
    
    if init_params is None:
        init_params = model.init_params

    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'model_class': model.__class__,
    'model_init_params': init_params
    }, model_path)
    print(f"Model saved as {model_path}")

def _modelversions_validate_load_params(mode,model_path_base,version,path):
    """
    Validiert die Parameter für das Laden eines Modells basierend auf dem angegebenen Modus.

    Diese Funktion überprüft die Gültigkeit der Eingabeparameter, die zum Laden eines Modells verwendet werden. 
    Je nach Modus werden unterschiedliche Parameter überprüft, um sicherzustellen, dass alle erforderlichen 
    Informationen bereitgestellt werden.

    Returns:
    bool: Gibt `True` zurück, wenn alle erforderlichen Parameter vorhanden und gültig sind, andernfalls `False`.
    """
    if mode in ['path'] and path is None:
        print('Please enter a valid path')
        return False
    if mode in ['version_fixed'] and version is None:
        print('Please enter a valid version')
        return False
    if mode in ['version_fixed','version_selection','latest'] and model_path_base is None:
        print('Please enter a valid model_path_base')
        return False
    return True

def _modelversions_get_loadpath(mode,model_path_base,version,path):
    """
    Bestimmt den Pfad zum Modell basierend auf dem angegebenen Modus und den Eingabeparametern.

    Diese Funktion verwendet den angegebenen Modus, um den richtigen Pfad zum Modell zu bestimmen, das geladen werden soll. 

    Parameters:
    mode (str): Der Modus, der angibt, wie das Modell geladen werden soll.

    model_path_base (str, optional): Der Basis-Pfad, unter dem das Modell gespeichert ist. 
                                     Dieser Parameter ist erforderlich für die Modi 'version_fixed', 'version_selection' und 'latest'.

    version (int, optional): Die Version des Modells, die geladen werden soll. 
                              Dieser Parameter ist erforderlich für den Modus 'version_fixed'.

    path (str, optional): Der vollständige Pfad zum Modell. 
                          Dieser Parameter ist erforderlich, wenn der Modus 'path' gewählt wurde.
    """
    if mode=='path':
        if not _modelversions_validate_load_params(mode,model_path_base,version,path):
            return None
        model_path=path
    elif mode=='version_fixed':
        if not _modelversions_validate_load_params(mode,model_path_base,version,path):
            return None
        model_path_base = _modelversions_get_model_path_with_version(model_path_base, version)
    elif mode=='version_selection':
        if not _modelversions_validate_load_params(mode,model_path_base,version,path):
            return None
        available_versions = _modelversions_get_available_versions(model_path_base)
        if available_versions:
            while True:
                try:
                    version = int(input(f"Enter the version to load ({_modelversions_format_versions(available_versions)} or 0 to cancel): "))
                    if version == 0:
                        print("Loading cancelled.")
                        return None
                    elif version in available_versions:
                        break
                    else:
                        print(f"Invalid version. Please enter a number out of {_modelversions_format_versions(available_versions)}, or 0 to cancel.")
                except ValueError:
                    print("Invalid input. Please enter a valid integer.")
        else:
            print("No versions available.")
            return None
        model_path = _modelversions_get_model_path_with_version(model_path_base, version)
    elif mode=='latest':
        if not _modelversions_validate_load_params(mode,model_path_base,version,path):
            return None
        available_versions = _modelversions_get_available_versions(model_path_base)
        if available_versions:
            version = available_versions[-1]
            model_path = _modelversions_get_model_path_with_version(model_path_base, version)
        else:
            print("No versions available.")
            return None
    else:
        print('Please enter a valid mode')
        return None
    return model_path


def modelversions_load_model(mode='latest',model_path_base=None,version=None,path=None):
    """
    Lädt ein Modell aus einer bestimmten Version oder einem angegebenen Pfad basierend auf dem Modus.

    Diese Funktion lädt ein gespeichertes Modell von einem bestimmten Pfad oder einer bestimmten Version, 
    indem sie den richtigen Pfad zum Modell ermittelt und das Modell sowie die zugehörigen Zustandsdaten (State Dict) 
    aus einer Datei lädt. Die Funktion erstellt eine Instanz des Modells mit den gespeicherten Initialisierungsparametern 
    und stellt den Modellzustand wieder her.

    Parameters:
    mode (str): Der Modus, in dem das Modell geladen werden soll. Bestimmt, wie der Pfad zum Modell ermittelt wird.
                Mögliche Werte sind:
                - 'path': Ein vollständiger Pfad zum Modell wird verwendet.
                - 'version_fixed': Eine spezifische Version des Modells wird verwendet.
                - 'version_selection': Der Benutzer wird zur Auswahl einer Version aufgefordert.
                - 'latest': Die neueste verfügbare Version des Modells wird verwendet.
                Standardwert ist 'latest'.

    model_path_base (str, optional): Der Basis-Pfad, unter dem das Modell gespeichert ist. 
                                     Erforderlich für die Modi 'version_fixed', 'version_selection' und 'latest'.

    version (int, optional): Die Version des Modells, die geladen werden soll. 
                              Erforderlich für den Modus 'version_fixed'.

    path (str, optional): Der vollständige Pfad zum Modell. 
                          Erforderlich, wenn der Modus 'path' gewählt wird.

    Returns:
    torch.nn.Module: Das geladene Modell als Instanz der Modellklasse.

    Example:
    >>> model = modelversions_load_model('latest', model_path_base='/path/to/models')
    Model loaded from /path/to/models_model_v{latest model}.pt
    
    >>> model = modelversions_load_model('version_fixed', model_path_base='/path/to/models', version=2)
    Model loaded from /path/to/models_model_v2.pt
    
    >>> model = modelversions_load_model('path', path='/path/to/models/model.pt')
    Model loaded from /path/to/models/model_v2.pt
    """
    model_path = _modelversions_get_loadpath(mode,model_path_base,version,path)
    if model_path is None:
        return None
    checkpoint = torch.load(model_path)
    model_class = checkpoint['model_class']
    model_init_params = checkpoint['model_init_params']
    
    # Initialisieren des Modells mit den gespeicherten Parametern
    model = model_class(**model_init_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {model_path}")
    return model




def train_calculate_class_weights(labels, method='sqrt'):
    """
    Berechnet die Gewichte für die Klassen basierend auf der Häufigkeit der Labels und der angegebenen Methode.

    Args:
    labels (torch.Tensor): Tensor der Labels. Dies ist ein eindimensionaler Tensor, der die Klassenzugehörigkeit 
                           jedes Beispiels enthält. Die Labels sollten ganzzahlige Werte sein.
    method (str): Methode zur Berechnung der Gewichte. Es stehen drei Methoden zur Verfügung:
                  'sqrt': Umgekehrt proportional zur Quadratwurzel der Klassenhäufigkeit.
                  'log': Logarithmische Gewichtung basierend auf der Klassenhäufigkeit.
                  'inverse': Umgekehrt proportional zur Klassenhäufigkeit.
                  Der Standardwert ist 'sqrt'.

    Returns:
    torch.Tensor: Tensor der Klassen-Gewichte. Jeder Eintrag im Tensor repräsentiert das Gewicht der entsprechenden Klasse.
                  Die Reihenfolge der Gewichte entspricht der Reihenfolge der eindeutigen Klassen, die in den Labels 
                  vorhanden sind.
    
    Raises:
    ValueError: Wenn die angegebene Methode nicht 'sqrt', 'log' oder 'inverse' ist.
    
    Beispiel:
    >>> labels = torch.tensor([0, 1, 1, 2, 2, 2])
    >>> calculate_class_weights(labels, method='inverse')
    tensor([3.0000, 1.5000, 1.0000])
    """

    # Berechne die Häufigkeit jeder Klasse
    unique_classes, class_counts = torch.unique(labels, return_counts=True)
    
    # Berechne die Gesamtzahl der Beispiele
    total_samples = labels.size(0)
    
    if method == 'sqrt':
        # Umgekehrt proportional zur Quadratwurzel der Klassenhäufigkeit
        weights = torch.sqrt(total_samples / class_counts.float())
    elif method == 'log':
        # Logarithmische Gewichtung
        weights = torch.log(1 + total_samples / class_counts.float())
    elif method == 'inverse':
        # Umgekehrt proportional zur Klassenhäufigkeit
        weights = total_samples / class_counts.float()
    else:
        raise ValueError("Method must be 'sqrt', 'log', or 'inverse'")
    
    # Normalisierung (optional)
    weights = weights / torch.sum(weights) * len(unique_classes)
    
    # Erstelle eine Gewichtungstabelle für alle möglichen Klassen
    class_weights = torch.zeros(len(unique_classes), dtype=torch.float)
    class_weights[unique_classes] = weights
    
    return class_weights  