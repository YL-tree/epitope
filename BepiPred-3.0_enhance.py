"""
使用BepiPred_3.0先对氨基酸序列进行预测，输出预测结果，作为DDPM框架的输入，进行预测增强
"""

### IMPORTS ###
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
from data import Antigens

### STATIC PATHS ###
ROOT_DIR = Path( Path(__file__).parent.resolve() )
MODELS_PATH = ROOT_DIR / "BP3Models"
#ESM_SCRIPT_PATH = ROOT_DIR / "extract.py"

### SET GPU OR CPU ###
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU device detected: {device}")
else:
    device = torch.device("cpu")
    print(f"GPU device not detected. Using CPU: {device}")

### MODEL ###

class MyDenseNetWithSeqLen(nn.Module):
    def __init__(self,
                 esm_embedding_size = 1281,
                 fc1_size = 150,
                 fc2_size = 120,
                 fc3_size = 45,
                 fc1_dropout = 0.7,
                 fc2_dropout = 0.7,
                 fc3_dropout = 0.7,
                 num_of_classes = 2):
        super(MyDenseNetWithSeqLen, self).__init__()
        
        
        self.esm_embedding_size = esm_embedding_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.fc3_size = fc3_size
        self.fc1_dropout = fc1_dropout
        self.fc2_dropout = fc2_dropout
        self.fc3_dropout = fc3_dropout
        
        self.ff_model = nn.Sequential(nn.Linear(esm_embedding_size, fc1_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc1_dropout),
                                      nn.Linear(fc1_size, fc2_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc2_dropout),
                                      nn.Linear(fc2_size, fc3_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc3_dropout),
                                      nn.Linear(fc3_size, num_of_classes))
    
    def forward(self, antigen):
        batch_size = antigen.size(0)
        seq_len = antigen.size(1)
        #convert dim (N, L, esm_embedding) --> (N*L, esm_embedding)
        output = torch.reshape(antigen, (batch_size*seq_len, self.esm_embedding_size))
        output = self.ff_model(output)                                               
        return output

class MyDenseNet(nn.Module):
    def __init__(self,
                 esm_embedding_size = 1280,
                 fc1_size = 150,
                 fc2_size = 120,
                 fc3_size = 45,
                 fc1_dropout = 0.7,
                 fc2_dropout = 0.7,
                 fc3_dropout = 0.7,
                 num_of_classes = 2):
        super(MyDenseNet, self).__init__()
        
        
        self.esm_embedding_size = esm_embedding_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.fc3_size = fc3_size
        self.fc1_dropout = fc1_dropout
        self.fc2_dropout = fc2_dropout
        self.fc3_dropout = fc3_dropout
        
        self.ff_model = nn.Sequential(nn.Linear(esm_embedding_size, fc1_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc1_dropout),
                                      nn.Linear(fc1_size, fc2_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc2_dropout),
                                      nn.Linear(fc2_size, fc3_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc3_dropout),
                                      nn.Linear(fc3_size, num_of_classes))
    
    def forward(self, antigen):
        batch_size = antigen.size(0)
        seq_len = antigen.size(1)
        #convert dim (N, L, esm_embedding) --> (N*L, esm_embedding)
        output = torch.reshape(antigen, (batch_size*seq_len, self.esm_embedding_size))
        output = self.ff_model(output)                                               
        return output

### CLASSES ###
class MyAntigens():
    def __init__(self, esm_files):
        self.esm_files = esm_files
        self.accs, self.seqs, self.esm_embeddings, self.epitopes = self.load_data()
        self.add_seq_len = True

    def load_data(self):
        esm_embeddings = []
        accs = []
        seqs = []
        epitopes = []
        for esm_file in self.esm_files:
            # 加载已经存在的pt文件
            data = torch.load(esm_file)
            esm_embedding = data['esm_representation']
            acc = data['acc']
            seq = data['seq']
            epitope = data['epitope']
            esm_embeddings.append(esm_embedding)
            accs.append(acc)
            seqs.append(seq)
            epitopes.append(epitope)
        return accs, seqs, esm_embeddings, epitopes
        

class BP3EnsemblePredict():
    
    def __init__(self, antigens, device = None, rolling_window_size = 9):
        """
        Inputs and initialization:
            antigens: Antigens class object
            device: pytorch device to use, default is cuda if available else cpu.
            
        """
        
        self.bp3_ensemble_run = False
        self.antigens = antigens
        self.rolling_window_size = rolling_window_size 

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        if antigens.add_seq_len:
            self.model_architecture = MyDenseNetWithSeqLen()
            m_path = MODELS_PATH / "BP3C50IDSeqLenFFNN" 
            self.model_states = list( m_path.glob("*Fold*") )
            self.classification_thresholds = {'Fold1': 0.21326530612244898,
                                              'Fold2': 0.15510204081632653,
                                              'Fold3': 0.1163265306122449,
                                              'Fold4': 0.09693877551020408,
                                              'Fold5': 0.19387755102040816}
            self.threshold_keys = [model_state.stem for model_state in self.model_states] 

        else:
            self.model_architecture = MyDenseNet()
            m_path = MODELS_PATH / "BP3C50IDFFNN" 
            self.model_states = list( m_path.glob("*Fold*") )
            self.classification_thresholds = {'Fold1': 0.2326530612244898,
                                              'Fold2': 0.15510204081632653,
                                              'Fold3': 0.1163265306122449,
                                              'Fold4': 0.15510204081632653,
                                              'Fold5': 0.19387755102040816}
            self.threshold_keys = [model_state.stem for model_state in self.model_states]


        #user specified classification thresholds for each fold
#        if classification_thresholds != None:
#            self.classification_thresholds = classification_thresholds


    def compute_rolling_mean_on_bp3_prob_outputs(self, antigen_avg_ensemble_probs):
        #same=ensures that the rolling mean output will have the same length as the number of residues for antigen.
        antigen_avg_ensemble_probs = antigen_avg_ensemble_probs.cpu().detach().numpy()
        return np.convolve(antigen_avg_ensemble_probs, np.ones(self.rolling_window_size), 'same') / self.rolling_window_size

    def run_bp3_ensemble(self):
        """
        INPUTS: antigens: Antigens() class object.  
        
        OUTPUTS:
                No outputs. Stores probabilities of ensemble models in Antigens() class object.
                Run bp3_pred_variable_threshold() or bp3_pred_majority_vote() afterwards to make predictions. 
        """
        num_of_models = len(self.model_states)
        ensemble_probs = list()
        softmax_function = nn.Softmax(dim=1)
        model = self.model_architecture

        data = list( zip(self.antigens.accs, self.antigens.seqs, self.antigens.esm_embeddings) )
        print("Generating BepiPred-3.0 scores")
        for acc, seq, esm_embedding in data:
            ensemble_prob = list()
            esm_encoding = esm_embedding.unsqueeze(0).to(self.device)
            
            for i in range(num_of_models):
                with torch.no_grad():
                
                    model_state = self.model_states[i] 
                    model.load_state_dict(torch.load(model_state, map_location=self.device))
                    model = model.to(self.device)
                    model.eval()
                    model_output = model(esm_encoding)
                    model_probs = softmax_function(model_output)[:, 1]

                    ensemble_prob.append(model_probs)

            ensemble_probs.append(ensemble_prob)
        
        self.bp3_ensemble_run = True
        self.antigens.ensemble_probs = ensemble_probs


    def create_csvfile(self, outfile_path):
        try:
            outfile_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Directory B-cell epitope predictions already there. Saving results there.")
        
        if not self.bp3_ensemble_run:
            sys.exit("BP3 ensemble has not been run, so predictions cannot be made.\
                Use method run_bp3_ensemble(antigens).")
        else:
            data = list(zip(self.antigens.accs, self.antigens.seqs, self.antigens.esm_embeddings, self.antigens.epitopes, self.antigens.ensemble_probs))
            
            for acc, seq, esm_embedding, epitope, ensemble_prob in data:
                # 为每个序列创建单独的pt文件
                pt_file = outfile_path / f"{acc}.pt"
                
                # 计算平均概率和滑动平均
                avg_prob = torch.mean(torch.stack(ensemble_prob, axis=1), axis=1)
                avg_prob_rolling_mean = self.compute_rolling_mean_on_bp3_prob_outputs(avg_prob)
                
                # 保存为pt文件
                torch.save({
                    'accession': acc,
                    'sequence': seq,
                    'esm_embedding': esm_embedding,
                    'epitope': epitope,
                    'avg_prob': avg_prob,
                    'rolling_mean': avg_prob_rolling_mean
                }, pt_file)
                
                print(f"Saved predictions for {acc} to {pt_file}")
            
    def bp3_pred_majority_vote(self, outfile_path):
        """
        
        """
        try:
            outfile_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Directory B-cell epitope predictions already there. Saving results there.")
        else:
            print("Directory B-cell epitope predictions not found. Made new one. ")
        
        if not self.bp3_ensemble_run:
            sys.exit("BP3 ensemble has not been run, so predictions cannot be made.\
 Use method run_bp3_ensemble(antigens).")
        else:
            data = list( zip(self.antigens.accs, self.antigens.seqs, self.antigens.ensemble_probs) )
            ensemble_preds = list()
            outfile_content = str()
            
            #go through each antigen
            for acc, seq, ensemble_prob in data:
                all_model_preds = list()
                num_residues = len(seq)
                
                #collect all predictions of all models in ensemble
                for i in range( len(ensemble_prob) ):
                    model_probs = ensemble_prob[i]
                    classification_threshold = self.classification_thresholds[ self.threshold_keys[i] ]
                    model_preds = [1 if res >= classification_threshold else 0 for res in model_probs]
                    all_model_preds.append(model_preds)
                    
                #ensemble majority vote 
                ensemble_pred = np.asarray(all_model_preds)
                ensemble_pred_len = np.shape(ensemble_pred)[1]

                if ensemble_pred_len < num_residues:
                    print(f"Sequence longer than what the ESM-2 trasnformer can encode entirely, {acc}. Outputting predictions up till {ensemble_pred_len} position.")
                
                ensemble_pred = [np.argmax( np.bincount(ensemble_pred[:, i]) ) for i in range(ensemble_pred_len)]
                epitope_preds = "".join([seq[i].upper() if ensemble_pred[i] == 1 else seq[i].lower() for i in range( len(ensemble_pred) )])
                outfile_content += f">{acc}\n{epitope_preds}\n"
                ensemble_preds.append(ensemble_pred)
            
            self.antigens.ensemble_preds = ensemble_preds
            outfile_content = outfile_content[:-1]
            #saving output to fasta formatted output file
            with open(outfile_path / "Bcell_epitope_preds.fasta", "w") as outfile:
                outfile.write(outfile_content)

if __name__ == "__main__":
    # 获取所有ESM编码文件路径列表
    esm_files = list(Path("data/esm_encodings").glob("*.pt"))
    pred = "mjv_pred"
    out_dir = Path("data/BepiPred_outputs")
    # 实例化MyAntigens类
    myAntigens = MyAntigens(esm_files)
    MyBP3EnsemblePredict = BP3EnsemblePredict(myAntigens)
    MyBP3EnsemblePredict.run_bp3_ensemble()
    MyBP3EnsemblePredict.create_csvfile(out_dir)

    #raw_ouput_and_top_epitope_candidates(out_dir, top_cands)
    ## B-cell epitope predictions ##
    if pred == "mjv_pred":
        MyBP3EnsemblePredict.bp3_pred_majority_vote(out_dir)
    elif pred == "vt_pred":
        MyBP3EnsemblePredict.bp3_pred_variable_threshold(out_dir, var_threshold=var_threshold)

