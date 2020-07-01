from torch.utils import data


class LifelongFewRelDataset(data.Dataset):

    def __init__(self, data, relation_names):
        self.relation_names = relation_names
        self.label = []
        self.candidate_relations = []
        self.text = []

        for entry in data:
            self.label.append(self.relation_names[entry[0]])
            negative_relations = entry[1]
            candidate_relation_names = [self.relation_names[x] for x in negative_relations]
            self.candidate_relations.append(candidate_relation_names)
            self.text.append(entry[2])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.text[index], self.label[index], self.candidate_relations[index]
