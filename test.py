import numpy as np
import pandas as pd
import pickle

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
            df[column + '_normalized'] = (df[column] - mean) / std
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
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__,
        'model_init_params': model.__dict__
    }, model_path)
    print(f"Model saved as {model_path}")


def modelversions_load_model(model_path_base):
    """
    Lädt ein Modell aus einer spezifischen Version.

    Parameters:
    model_path_base (str): Der Basispfad, unter dem das Modell gespeichert ist.

    Returns:
    torch.nn.Module: Das geladene Modell oder None, falls der Ladevorgang abgebrochen wurde.

    Example:
    >>> model = modelversions_load_model('/path/to/models/test')
    Available versions: [1, 2, 3]
    Enter 0 to cancel.
    Enter the version to load (1-3 or 0 to cancel): 2
    Model loaded from /path/to/models/test_v2.pt
    """
    available_versions = _modelversions_get_available_versions(model_path_base)
    if not available_versions:
        print("No versions available.")
        return None

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

    
    model_path = _modelversions_get_model_path_with_version(model_path_base, version)
    checkpoint = torch.load(model_path)
    model_class = checkpoint['model_class']
    model_init_params = checkpoint['model_init_params']
    model = model_class(**model_init_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {model_path}")
    return model

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

















# Create or load model
model = mytools.modelversions_load_model(model_path_base)
if model is None:
    # If no model was loaded, initialize a new one
    model = MultilingualBERTClassifier(num_additional_features=num_additional_features, num_classes=num_classes)
    print('New model created.')
else:
    print('Model loaded from saved version.')

# Move model to device if possible
model.to(device)
print('Model on device')