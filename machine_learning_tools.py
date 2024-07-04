import pickle

class ExtendedLabelEncoder:
    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = {}
        self.num_classes = 0
    
    def fit(self, labels):
        unique_labels = set(labels)
        self.num_classes = len(unique_labels)
        
        for index, label in enumerate(unique_labels):
            self.label_to_index[label] = index
            self.index_to_label[index] = label
    
    def transform(self, labels):
        return [self.label_to_index[label] for label in labels]
    
    def inverse_transform(self, indices):
        return [self.index_to_label[index] for index in indices]
    
    def transform_fit(self, labels):
        self.fit(labels)
        return self.transform(labels)
    
    def extend(self, new_labels):
        existing_labels = set(self.label_to_index.keys())
        for label in new_labels:
            if label not in existing_labels:
                self.label_to_index[label] = len(self.label_to_index)
                self.index_to_label[len(self.index_to_label)] = label
                existing_labels.add(label)
        self.num_classes = len(self.label_to_index)
    
    def extend_transform(self, new_labels):
        self.extend(new_labels)
        return self.transform(new_labels)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            encoder = pickle.load(f)
        self.label_to_index = encoder.label_to_index
        self.index_to_label = encoder.index_to_label
        self.num_classes = encoder.num_classes
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    def __len__(self):
        return self.num_classes
    
    def labels(self):
        return {
            'label_to_index': self.label_to_index,
            'index_to_label': self.index_to_label
        }