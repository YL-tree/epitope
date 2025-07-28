import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
import esm
import torch
from torch.utils.data import Dataset, DataLoader
import torch

def map_to_infinite_range(x, mu=0, sigma=1):
    # 处理边界情况，避免产生无限值
    eps = torch.finfo(torch.float).eps
    x = torch.clamp(x, eps, 1-eps)
    
    return torch.erfinv(2 * x - 1) * torch.sqrt(torch.tensor(2.)) * sigma + mu


class Antigens():
    """
    用于处理抗原序列数据的类。
    
    属性:
        sequence_folder (Path): 包含氨基酸序列txt文件的文件夹路径。
        antigen_folder (Path): 包含抗原信息csv文件的文件夹路径。
        esm_encoding_dir (Path): 用于存储ESM-2编码的文件夹路径。

        add_seq_len (bool): 是否添加序列长度特征。
        antigens (pd.DataFrame): 包含抗原信息的DataFrame。
        seqs (list): 包含氨基酸序列的列表。
        accs (list): 包含抗原访问号的列表。
        esm_encoding_paths (list): 包含ESM-2编码文件路径的列表。
    """
    def __init__(self, sequence_folder, antigen_folder, esm_encoding_dir,
        add_seq_len=False):
        self.sequence_folder = sequence_folder
        self.antigen_folder = antigen_folder
        self.esm_encoding_dir = esm_encoding_dir
        self.epitopes, self.seqs, self.accs = self.process_data()
        self.add_seq_len = add_seq_len
        self.esm_encoding_paths = self.get_esm2_represention_on_accs_seqs()
        
        num_of_seqs = len(self.seqs)
        print(f"Number of sequences detected in fasta file: {num_of_seqs}")
        print(f"ESM-2 encoding sequences. Saving encodings to {str(esm_encoding_dir)}")

        try:
            esm_encoding_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Directory for esm encodings already there. Saving encodings there.")
        else:
            print("Directory for esm encodings not found. Made new one.")


    def extract_response_frequency_segments(self, df, column='response frequency', max_length=500):
        """
        从DataFrame中提取响应频率有值的片段
        
        参数:
            df: 包含数据的DataFrame
            column: 包含响应频率的列名（默认值为'response frequency'）
            max_length: 每个片段的最大长度（默认值为500）
        
        返回:
            包含响应频率有值片段的DataFrame列表
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        # 创建有效标记列
        valid_mask = df[column].notna()
        
        # 使用.loc避免SettingWithCopyWarning
        df = df.copy()  # 创建副本以避免修改原DataFrame
        df.loc[:, 'segment_id'] = (valid_mask & ~valid_mask.shift(1, fill_value=False)).cumsum()
        
        # 筛选有效行并按segment_id分组
        segments = []
        for _, group in df[valid_mask].groupby('segment_id'):
            group = group.drop(columns=['segment_id'])
            
            # 检查长度是否超过限制
            if len(group) > max_length:
                # 按max_length大小拆分
                for i in range(0, len(group), max_length):
                    segments.append(group.iloc[i:i+max_length])
            else:
                segments.append(group)
        
        return segments

    def process_data(self):
        """
        处理数据并返回预处理结果
        
        参数:
            sequence_folder: 包含序列txt文件的文件夹路径
            antigen_folder: 包含抗原csv文件的文件夹路径
        
        返回:
            epitope_list: 响应频率列表
            amino_list: 氨基酸序列列表
            position_list: 位置信息列表
            sequences_length: 序列长度列表
            acc: 序列名称列表
        """
        # 提取sequence数据
        all_sequence = {}
        files = os.listdir(self.sequence_folder)
        for file in files:
            file_path = os.path.join(self.sequence_folder, file)
            if file.endswith('.txt'):
                filename_without_extension = os.path.splitext(file)[0]
                with open(file_path, 'r') as f:
                    data = f.read()
                    data = pd.DataFrame({'antigen': list(data)})
                    all_sequence[filename_without_extension] = data
        
        # 提取epitope数据
        all_epitope_curve = {}
        all_amino_position = {}
        files = os.listdir(self.antigen_folder)
        for file in files:
            file_path = os.path.join(self.antigen_folder, file, 'epitope-curve.csv')
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                all_epitope_curve[file] = data['response frequency']
                all_amino_position[file] = data['position']
        
        # 合并数据
        merged_sequence = {}
        for key in all_sequence:
            if key in all_epitope_curve:
                merged_sequence[key] = pd.concat([
                    all_sequence[key],
                    all_epitope_curve[key], 
                    all_amino_position[key]
                ], axis=1, ignore_index=False)

        # 预处理数据
        epitope_list = []
        amino_list = []
        accs = []

            
        for acc, value in merged_sequence.items():
            # 提取response frequency有值的片段
            # 如果有多个片段有值，则切割为多个片段
            segments = self.extract_response_frequency_segments(value)
            for idx, segment in enumerate(segments):
                segment = segment.dropna(subset=['response frequency'])
                # 当长度大于5时，保存数据
                if len(segment) > 5:
                    epitope_list.append(segment['response frequency'].tolist())
                    amino_list.append(segment['antigen'].tolist())
                    # position_list.append(segment['position'].tolist())
                    # sequences_length.append(len(segment['antigen']))
                    accs.append(acc + '_' + str(idx))

        self.check_accepted_AAs(accs, amino_list)
        # return x_list, y_list, position_list, sequences_length, accs
        return epitope_list, amino_list, accs

    def check_accepted_AAs(self, accs, sequences):
        accepted_AAs = set(["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"])
        entries = list( zip(accs, sequences) ) 
    
        for entry in entries:
            acc = entry[0]
            seq = entry[1]
            #if not accessions or sequences obtained (empty)
            check = all(res.upper() in accepted_AAs for res in seq)
            if not check:
                sys.exit(f"Nonstandard amino acid character detected in acc: {acc}. Allowed character lower and uppercase amino acids:\n{accepted_AAs}")

    def tuple_generator(self, data, batch_size=10):
        length = len(data)
        for i in range(0, length, batch_size):
            yield data[i:i + batch_size]

    def get_esm2_represention_on_accs_seqs(self):
        """
        data: list of tuples: [(seq_name, sequence)...]
        per_res_representations:
        """

        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        #esm_representations = []
        #preparing batch for ESM2
        upper_case_sequences = [''.join(s).upper() for s in self.seqs]  # 修改这一行
        data = list(zip(self.accs, upper_case_sequences, self.epitopes))
        nr_seqs = len(data)
        batch_generator = self.tuple_generator(data)
        
        enc_id = 0
        enc_paths = []

        for b in batch_generator:
            batch_data = [(item[0], item[1]) for item in b]
            seqs = [item[1] for item in b]
            epitopes = [item[2] for item in b]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            acc_names = batch_labels
            # epitopes = b[2]

            # Extract per-residue representations
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
            token_representations = results["representations"][33]
    
            #encoding sequences
            for i, tokens_len in enumerate(batch_lens):

                esm_representation = token_representations[i, 1 : tokens_len - 1]
                if self.add_seq_len:
                    esm_representation = self.add_seq_len_feature(esm_representation)

                enc_path = self.esm_encoding_dir / f"{acc_names[i]}.pt"
                epitope = torch.tensor(epitopes[i], dtype=torch.float32)
                # print(epitope.shape)
                embedding_data = {
                    'acc': acc_names[i],
                    'seq': seqs[i],
                    'epitope': epitope,
                    'esm_representation': esm_representation
                }
                # 将acc, seq, epitope, esm_representation均保存进pt文件
                torch.save(embedding_data, enc_path)

                enc_paths.append(enc_path)
                enc_id += 1

                print(f"ESM-2 encoded sequence {acc_names[i]} {enc_id}/{nr_seqs}")

        return enc_paths

    def add_seq_len_feature(self, X):
        #adding sequence length to each positional ESM-2 embedding
        seq_len = X.size()[0]
        seq_len_v = torch.ones(seq_len)*seq_len
        seq_len_v = seq_len_v.unsqueeze(dim=1)
        new_X = torch.cat((X, seq_len_v), axis=1)
        
        return new_X

class ESM2Dataset(Dataset):
    def __init__(self, esm_files):
        self.esm_files = esm_files
        self.accs, self.esm_embeddings, self.epitope_labels = self.load_data()

    def load_data(self):
        esm_embeddings = []
        epitope_labels = []
        accs = []
        for esm_file in self.esm_files:
            data = torch.load(esm_file, weights_only=False)
            acc = data['acc']
            esm_embedding = data['esm_representation']
            epitope_label = data['epitope']
            epitope_label = map_to_infinite_range(epitope_label)
            accs.append(acc)
            esm_embeddings.append(esm_embedding)
            epitope_labels.append(epitope_label)
        return accs, esm_embeddings, epitope_labels

    def __len__(self):
        return len(self.esm_files)

    def __getitem__(self, idx):
        acc = self.accs[idx]
        esm_embedding = self.esm_embeddings[idx]
        epitope_label = self.epitope_labels[idx]
        return acc, esm_embedding, epitope_label

class BepiPredDataset(Dataset):
    def __init__(self, bepi_files):
        self.bepi_files = bepi_files
        self.accs, self.esm_embeddings, self.epitope_labels, self.bepi_pred = self.load_data()

    def load_data(self):
        accs = []
        esm_embeddings = []
        epitope_labels = []
        bepi_preds = []
        for bepi_file in self.bepi_files:
            data = torch.load(bepi_file, weights_only=False)
            acc = data['accession']
            esm_embedding = data['esm_embedding']
            epitope_label = data['epitope']
            bepi_pred = data['avg_prob']
            accs.append(acc)
            esm_embeddings.append(esm_embedding)
            epitope_labels.append(epitope_label)
            bepi_preds.append(bepi_pred)
        return accs, esm_embeddings, epitope_labels, bepi_preds

    def __len__(self):
        return len(self.bepi_files)

    def __getitem__(self, idx):
        acc = self.accs[idx]
        esm_embedding = self.esm_embeddings[idx]
        epitope_label = self.epitope_labels[idx]
        bepi_pred = self.bepi_pred[idx]
        return acc, esm_embedding, epitope_label, bepi_pred



if __name__ == "__main__":
    sequence_folder = Path("data/sequence")
    antigen_folder = Path("data/antigens")
    esm_encoding_dir = Path("data/esm_encodings")
    
    antigens = Antigens(sequence_folder, antigen_folder, esm_encoding_dir, add_seq_len=True)


    enc_paths = antigens.get_esm2_represention_on_accs_seqs()
