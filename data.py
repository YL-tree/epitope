import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
import esm
import torch


class Antigens():
    def __init__(self, sequence_folder, antigen_folder, esm_encoding_dir,
        add_seq_len=False):
        self.sequence_folder = sequence_folder
        self.antigen_folder = antigen_folder
        self.esm_encoding_dir = esm_encoding_dir
        self.antigens, self.seqs, self.accs = self.process_data()
        self.add_seq_len = add_seq_len
        
        num_of_seqs = len(self.seqs)
        print(f"Number of sequences detected in fasta file: {num_of_seqs}")
        print(f"ESM-2 encoding sequences. Saving encodings to {str(esm_encoding_dir)}")

        try:
            esm_encoding_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Directory for esm encodings already there. Saving encodings there.")
        else:
            print("Directory for esm encodings not found. Made new one.")


    def process_data(self):
        """
        处理数据并返回预处理结果
        
        参数:
            sequence_folder: 包含序列txt文件的文件夹路径
            antigen_folder: 包含抗原csv文件的文件夹路径
        
        返回:
            x_list: 响应频率列表
            y_list: 抗原序列列表
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
        x_list = []
        y_list = []
        # position_list = []
        # sequences_length = []
        accs = []
        for acc, value in merged_sequence.items():
            y_list.append(value["antigen"])
            x_list.append(value['response frequency'])
            # position_list.append(value['position'])
            # sequences_length.append(len(value))
            accs.append(acc)
        
        self.check_accepted_AAs(accs, y_list)
        # return x_list, y_list, position_list, sequences_length, accs
        return x_list, y_list, accs

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
        upper_case_sequences = [''.join(s.tolist()).upper() for s in self.seqs]  # 修改这一行
        data = list(zip(self.accs, upper_case_sequences))
        nr_seqs = len(data)
        batch_generator = self.tuple_generator(data)
        
        enc_id = 0
        enc_paths = []

        for b in batch_generator:

            batch_labels, batch_strs, batch_tokens = batch_converter(b)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            acc_names = batch_labels

            # Extract per-residue representations
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
            token_representations = results["representations"][33]
    
            #encoding sequences
            for i, tokens_len in enumerate(batch_lens):

                esm_representation = token_representations[i, 1 : tokens_len - 1]
                if self.add_seq_len:
                    esm_representation = self.add_seq_len_feature(esm_representation)

                enc_path = self.esm_encoding_dir / f"{acc_names[i]}_{enc_id}.pt"
                torch.save(esm_representation, enc_path)

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

if __name__ == "__main__":
    sequence_folder = Path("data/sequence")
    antigen_folder = Path("data/antigens")
    esm_encoding_dir = Path("data/esm_encodings")
    antigens = Antigens(sequence_folder, antigen_folder, esm_encoding_dir, add_seq_len=True)
    enc_paths = antigens.get_esm2_represention_on_accs_seqs()
