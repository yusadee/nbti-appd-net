from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rc('font', family='serif', size=12)
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cpu')


def default_config(script_dir: Path) -> Dict[str, Any]:
    return {
        'data': {
            'base_path': str(script_dir),
            'file1_name': 'vgs=4.csv',
            'file2_name': 'T=150.csv',
            'keyword': 'vthgmlist',
            'min_time': 1.0,
        },
        'output': {
            'show_fig': True,
            'save_fig': True,
            'figure_dir': 'figures_exp156_pretrained',
        },
    }


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)



def maybe_save_show(fig, filename: Optional[str], cfg_output: Dict[str, Any]):
    if cfg_output['save_fig'] and filename is not None:
        ensure_dir(Path(cfg_output['figure_dir']))
        fig.savefig(Path(cfg_output['figure_dir']) / filename, bbox_inches='tight', dpi=300)

    if not cfg_output['show_fig']:
        plt.close(fig)



def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mask = y_true.flatten() > 1e-6
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100 if np.any(mask) else 0.0
    r2 = r2_score(y_true, y_pred)
    return {'rmse': float(rmse), 'mse': float(mse), 'mape': float(mape), 'r2': float(r2)}


def dict_to_scaler(obj: Optional[Dict[str, Any]]) -> Optional[StandardScaler]:
    if obj is None:
        return None
    scaler = StandardScaler()
    scaler.mean_ = np.array(obj['mean_'], dtype=float)
    scaler.scale_ = np.array(obj['scale_'], dtype=float)
    scaler.var_ = np.array(obj['var_'], dtype=float)
    scaler.n_features_in_ = int(obj['n_features_in_'])
    scaler.n_samples_seen_ = 1
    return scaler


# =============================================================================
# Data
# =============================================================================
def parse_csv_data_final(file_path: Path, keyword: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f'File not found: {file_path}')
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    lines_lower = [line.lower().strip() for line in lines]
    keyword = keyword.lower()
    try:
        start_idx = [i for i, s in enumerate(lines_lower) if keyword in s][0]
    except IndexError as exc:
        raise ValueError(f"Keyword '{keyword}' was not found in file {file_path.name}.") from exc
    header_line_idx, data_start_idx = start_idx + 1, start_idx + 2
    header = [h.strip() for h in lines[header_line_idx].strip().split(',')]
    end_idx = data_start_idx
    while end_idx < len(lines) and lines[end_idx].strip() and not lines[end_idx].startswith(','):
        end_idx += 1
    data_lines = [line.strip().split(',') for line in lines[data_start_idx:end_idx]]
    df = pd.DataFrame(data_lines)
    num_cols = min(len(header), df.shape[1])
    df = df.iloc[:, :num_cols]
    df.columns = header[:num_cols]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(how='all', inplace=True)
    return df


def format_data_final(df, param_name, condition_name, const_condition_name, const_condition_value):
    try:
        time_col = [col for col in df.columns if 'time' in str(col).lower()][0]
    except IndexError as exc:
        raise ValueError("No column named 'TimeList' or containing 'Time' was found in the DataFrame.") from exc
    condition_cols = [col for col in df.columns if col != time_col]
    df_melted = df.melt(id_vars=[time_col], value_vars=condition_cols, var_name=condition_name, value_name=param_name)
    df_melted[condition_name] = df_melted[condition_name].astype(str).str.replace('_s', '', regex=False).astype(float)
    df_melted.rename(columns={time_col: 'Time'}, inplace=True)
    df_melted[const_condition_name] = const_condition_value
    return df_melted


def load_and_prepare_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    base_path = Path(cfg['data']['base_path'])
    file1_path = base_path / cfg['data']['file1_name']
    file2_path = base_path / cfg['data']['file2_name']
    keyword = cfg['data']['keyword']
    min_time = cfg['data']['min_time']
    vth1 = parse_csv_data_final(file1_path, keyword)
    vth2 = parse_csv_data_final(file2_path, keyword)
    vth1_long = format_data_final(vth1, 'dVth', 'Temperature', 'Vgs', 4.0)
    vth2_long = format_data_final(vth2, 'dVth', 'Vgs', 'Temperature', 150.0)
    full_df = pd.concat([vth1_long, vth2_long], ignore_index=True).drop_duplicates()
    full_df = full_df[full_df['Time'] > min_time].dropna().reset_index(drop=True)
    full_df[['dVth', 'Time']] = full_df[['dVth', 'Time']].astype(float)
    return full_df


def create_sliding_window_samples(df, window_size=5, max_time=None):
    all_windows = []
    df_filtered = df if max_time is None else df[df['Time'] <= max_time]
    for (temp, vgs), group in df_filtered.groupby(['Temperature', 'Vgs']):
        sorted_group = group.sort_values('Time')
        times = sorted_group['Time'].values.reshape(-1, 1)
        vths = sorted_group['dVth'].values.reshape(-1, 1)
        if len(times) < window_size:
            continue
        for i in range(len(times) - window_size + 1):
            all_windows.append({
                'conditions': np.array([temp, vgs], dtype=float),
                'time_series': times[i:i + window_size],
                'vth_series': vths[i:i + window_size],
            })
    return all_windows


def subset_windows_by_indices(windows_list: list, idx_arr: np.ndarray):
    return [windows_list[i] for i in idx_arr]


def prepare_point_data(df: pd.DataFrame, idx: np.ndarray):
    sub = df.iloc[idx]
    return (
        sub[['Temperature', 'Vgs']].values,
        sub[['Time']].values,
        sub[['dVth']].values,
    )


# =============================================================================
# =============================================================================
class ANN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class APPD_Net(nn.Module):
    def __init__(self, input_dim=2, shared_hidden_dim=64, A_head_dim=32, n_head_dim=32, attention_hidden_dim=16):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_dim), nn.Tanh(),
            nn.Linear(shared_hidden_dim, shared_hidden_dim), nn.Tanh(),
        )
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim), nn.Tanh(),
            nn.Linear(attention_hidden_dim, input_dim), nn.Softmax(dim=1),
        )
        self.A_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, A_head_dim), nn.ReLU(),
            nn.Linear(A_head_dim, 1), nn.Softplus(),
        )
        self.n_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, n_head_dim), nn.ReLU(),
            nn.Linear(n_head_dim, 1), nn.Sigmoid(),
        )

    def forward(self, x_cond):
        attention_weights = self.attention_layer(x_cond)
        x_attended = x_cond * attention_weights
        shared_features = self.shared_layers(x_attended)
        A = self.A_head(shared_features)
        sigmoid_output_n = self.n_head(shared_features)
        min_n = 1.0 / 6.0
        max_n = 1.0 / 3.0
        n = min_n + (max_n - min_n) * sigmoid_output_n
        return A, n, attention_weights

    def predict_dvth(self, x_cond, t):
        A, n, _ = self.forward(x_cond)
        return A * (t ** n)


class CNN1D(nn.Module):
    def __init__(self, in_channels=3, seq_len=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(64 * seq_len, seq_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x.view(x.size(0), -1, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64, dropout=0.2):
        super().__init__()
        self.time_embedding = nn.Linear(1, d_model)
        self.vth_embedding = nn.Linear(1, d_model)
        self.condition_embedding = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers,
            dim_feedforward, dropout, batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src_time, tgt_vth, conditions, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_emb = self.pos_encoder(self.time_embedding(src_time))
        tgt_emb = self.pos_encoder(self.vth_embedding(tgt_vth))
        cond_emb = self.condition_embedding(conditions).unsqueeze(1).repeat(1, src_time.size(1), 1)
        cond_emb_tgt = cond_emb[:, :tgt_vth.size(1), :]
        src_with_cond = src_emb + cond_emb
        tgt_with_cond = tgt_emb + cond_emb_tgt
        output = self.transformer(
            src_with_cond, tgt_with_cond,
            src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.fc_out(output)


# =============================================================================
# Prediction utilities
# =============================================================================
def load_ann(bundle_path: Path):
    bundle = torch.load(bundle_path, map_location='cpu')
    model = ANN()
    model.load_state_dict(bundle['state_dict'])
    model.eval()
    scaler = dict_to_scaler(bundle.get('scaler'))
    return model, scaler, bundle


def load_appd(bundle_path: Path):
    bundle = torch.load(bundle_path, map_location='cpu')
    model = APPD_Net()
    model.load_state_dict(bundle['state_dict'])
    model.eval()
    scaler = dict_to_scaler(bundle.get('scaler'))
    return model, scaler, bundle


def load_cnn(bundle_path: Path):
    bundle = torch.load(bundle_path, map_location='cpu')
    seq_len = bundle.get('window_size', 5)
    model = CNN1D(seq_len=seq_len)
    model.load_state_dict(bundle['state_dict'])
    model.eval()
    scaler = dict_to_scaler(bundle.get('scaler'))
    return model, scaler, bundle


def load_transformer(bundle_path: Path):
    bundle = torch.load(bundle_path, map_location='cpu')
    model = TimeSeriesTransformer()
    model.load_state_dict(bundle['state_dict'])
    model.eval()
    return model, bundle


def predict_cnn_curve(model_cpu, scaler, cond_df: pd.DataFrame, window_size: int):
    windows = create_sliding_window_samples(cond_df, window_size=window_size)
    if not windows:
        return np.full((len(cond_df), 1), np.nan, dtype=float)
    X_list = []
    for item in windows:
        _temp, _vgs = item['conditions']
        time_col = item['time_series']
        temp_col = np.full_like(time_col, _temp)
        vgs_col = np.full_like(time_col, _vgs)
        X_list.append(np.hstack([time_col, temp_col, vgs_col]))
    X = np.array(X_list)
    X_scaled = scaler.transform(X.reshape(-1, 3)).reshape(X.shape)
    with torch.no_grad():
        y_win = model_cpu(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
    N = len(cond_df)
    y_full = np.full((N, 1), np.nan, dtype=float)
    counts = np.zeros((N, 1), dtype=float)
    for i, w_pred in enumerate(y_win):
        y_full[i:i + window_size] = np.nansum(np.dstack([y_full[i:i + window_size], w_pred]), axis=2)
        counts[i:i + window_size] += 1
    counts[counts == 0] = 1
    return y_full / counts


def predict_transformer_curve(model_cpu, cond_df: pd.DataFrame, boundary: float):
    cond_df = cond_df.sort_values('Time').reset_index(drop=True)
    t = cond_df[['Time']].values.astype(float)
    T = float(cond_df['Temperature'].iloc[0])
    V = float(cond_df['Vgs'].iloc[0])
    N = len(cond_df)
    pred = np.full((N, 1), np.nan, dtype=float)
    known_mask = (cond_df['Time'].values < boundary)
    known_len = int(np.sum(known_mask))

    src_time = torch.tensor(t, dtype=torch.float32).unsqueeze(0)
    conditions = torch.tensor([[T, V]], dtype=torch.float32)

    if known_len >= 1:
        known_vth = torch.tensor(cond_df.loc[known_mask, 'dVth'].values, dtype=torch.float32).reshape(1, -1, 1)
        if known_len == 1:
            decoder_in = torch.zeros(1, 1, 1)
        else:
            decoder_in = torch.cat([torch.zeros(1, 1, 1), known_vth[:, :-1, :]], dim=1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_in.size(1))
        with torch.no_grad():
            out_known = model_cpu(src_time[:, :known_len, :], decoder_in, conditions, tgt_mask=tgt_mask)
        pred[:known_len, 0] = out_known[0, :, 0].numpy()

    if known_len >= 1:
        decoder_input = torch.cat([
            torch.zeros(1, 1, 1),
            torch.tensor(cond_df.loc[known_mask, 'dVth'].values, dtype=torch.float32).reshape(1, -1, 1),
        ], dim=1)
    else:
        decoder_input = torch.zeros(1, 1, 1)

    for k in range(max(known_len, 0), N):
        with torch.no_grad():
            current_src = src_time[:, :k + 1, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.size(1))
            out = model_cpu(current_src, decoder_input, conditions, tgt_mask=tgt_mask)
            next_val = out[:, -1, :].unsqueeze(1)
        pred[k, 0] = float(next_val[0, 0, 0].numpy())
        decoder_input = torch.cat([decoder_input, next_val], dim=1)
    return pred


def ensemble_mean_ci(pred_matrix: np.ndarray, alpha: float):
    mean = np.nanmean(pred_matrix, axis=0)
    low_q = 100.0 * (alpha / 2.0)
    high_q = 100.0 * (1.0 - alpha / 2.0)
    low = np.nanpercentile(pred_matrix, low_q, axis=0)
    high = np.nanpercentile(pred_matrix, high_q, axis=0)
    return mean, low, high


def interp_loglog_1d(x, y, x0):
    x = np.asarray(x).astype(float).reshape(-1)
    y = np.asarray(y).astype(float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return np.nan
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x0 <= x[0]:
        return float(y[0])
    if x0 >= x[-1]:
        return float(y[-1])
    j = np.searchsorted(x, x0)
    x1, x2 = x[j - 1], x[j]
    y1, y2 = y[j - 1], y[j]
    if x1 <= 0 or x2 <= 0 or y1 <= 0 or y2 <= 0:
        return float(y1 + (y2 - y1) * (x0 - x1) / (x2 - x1 + 1e-30))
    lx1, lx2, lx0 = np.log(x1), np.log(x2), np.log(x0)
    ly1, ly2 = np.log(y1), np.log(y2)
    ly0 = ly1 + (ly2 - ly1) * (lx0 - lx1) / (lx2 - lx1 + 1e-30)
    return float(np.exp(ly0))


def ensure_boundary_point(time_arr, series_dict, boundary, atol=1e-9):
    t = np.asarray(time_arr).astype(float).reshape(-1)
    if np.any(np.isclose(t, boundary, atol=atol, rtol=0.0)):
        is_bp = np.zeros_like(t, dtype=int)
        is_bp[np.argmin(np.abs(t - boundary))] = 1
        return t, series_dict, is_bp
    idx = int(np.searchsorted(t, boundary))
    t_new = np.insert(t, idx, boundary)
    is_bp = np.zeros_like(t_new, dtype=int)
    is_bp[idx] = 1
    series_new = {}
    for k, y in series_dict.items():
        y = np.asarray(y).astype(float).reshape(-1)
        y0 = interp_loglog_1d(t, y, boundary)
        series_new[k] = np.insert(y, idx, y0)
    return t_new, series_new, is_bp


def compute_zoom_ylim(t, series_list, boundary, low_margin=0.90, high_margin=1.10):
    t = np.asarray(t).reshape(-1)
    mask = (t >= boundary) & np.isfinite(t)
    vals = []
    for s in series_list:
        s = np.asarray(s).reshape(-1)
        v = s[mask]
        v = v[np.isfinite(v)]
        v = v[v > 0]
        if v.size > 0:
            vals.append(v)
    if len(vals) == 0:
        return None
    vv = np.concatenate(vals, axis=0)
    ymin = float(np.min(vv))
    ymax = float(np.max(vv))
    if ymin <= 0 or ymax <= 0:
        return None
    ymin_adj = ymin * low_margin
    ymax_adj = ymax * high_margin
    if ymax_adj <= ymin_adj:
        ymax_adj = ymin_adj * 1.2
    return ymin_adj, ymax_adj


# =============================================================================
# Plotting
# =============================================================================
def plot_single_accuracy(y_true, y_pred, metrics, model_name, filename, runtime, cfg_output):
    fig, ax = plt.subplots(figsize=(10, 8))
    text_metrics = (
        f"RMSE = {metrics['rmse']:.2f} mV\n"
        f"MSE = {metrics['mse']:.2f}\n"
        f"MAPE = {metrics['mape']:.2f} %\n"
        f"R2 = {metrics['r2']:.4f}\n"
        f"Loaded Runtime = {runtime:.4f} s"
    )
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', label='Test Set Samples')
    ax.plot([np.min(y_true), np.max(y_true)], [np.min(y_true), np.max(y_true)], 'r--', lw=2, label='y=x (Ideal)')
    ax.set_xlabel('True dVth (mV)')
    ax.set_ylabel('Predicted dVth (mV)')
    ax.set_title(f'{model_name}: Predicted vs. True (Loaded Weights)')
    ax.legend(loc='lower right')
    ax.grid(True)
    ax.text(0.05, 0.95, text_metrics, transform=ax.transAxes, fontsize=12, va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.tight_layout()
    maybe_save_show(fig, filename, cfg_output)


# =============================================================================
# Experiment 1
# =============================================================================
def run_exp1_from_weights(full_df: pd.DataFrame, weights_dir: Path, cfg_output: Dict[str, Any]):
    exp_dir = weights_dir / 'exp1'
    split_bundle = torch.load(exp_dir / 'split.pt', map_location='cpu')
    test_data_point = prepare_point_data(full_df, np.array(split_bundle['test_point_idx'], dtype=int))
    all_windows = create_sliding_window_samples(full_df, window_size=int(split_bundle['window_size']))
    test_windows = subset_windows_by_indices(all_windows, np.array(split_bundle['test_window_idx'], dtype=int))

    ann_model, ann_scaler, ann_bundle = load_ann(exp_dir / 'ann.pt')
    appd_model, appd_scaler, appd_bundle = load_appd(exp_dir / 'appd.pt')
    cnn_model, cnn_scaler, cnn_bundle = load_cnn(exp_dir / 'cnn.pt')
    trans_model, trans_bundle = load_transformer(exp_dir / 'transformer.pt')

    # ANN
    X_ann = ann_scaler.transform(np.hstack([test_data_point[0], test_data_point[1]]))
    with torch.no_grad():
        y_pred_ann = ann_model(torch.tensor(X_ann, dtype=torch.float32)).numpy()
    metrics_ann = calculate_metrics(test_data_point[2], y_pred_ann)

    # APPD
    X_appd = appd_scaler.transform(test_data_point[0])
    with torch.no_grad():
        y_pred_appd = appd_model.predict_dvth(torch.tensor(X_appd, dtype=torch.float32), torch.tensor(test_data_point[1], dtype=torch.float32)).numpy()
    metrics_appd = calculate_metrics(test_data_point[2], y_pred_appd)

    # CNN
    def prepare_cnn_data(data_seq):
        X, y = [], []
        for item in data_seq:
            temp, vgs = item['conditions']
            time_col = item['time_series']
            vth_col = item['vth_series']
            temp_col = np.full_like(time_col, temp)
            vgs_col = np.full_like(time_col, vgs)
            X.append(np.hstack([time_col, temp_col, vgs_col]))
            y.append(vth_col)
        return np.array(X), np.array(y)
    X_cnn, y_cnn = prepare_cnn_data(test_windows)
    X_cnn_scaled = cnn_scaler.transform(X_cnn.reshape(-1, 3)).reshape(X_cnn.shape)
    with torch.no_grad():
        y_pred_cnn = cnn_model(torch.tensor(X_cnn_scaled, dtype=torch.float32)).numpy().reshape(-1, 1)
    y_test_cnn = y_cnn.reshape(-1, 1)
    metrics_cnn = calculate_metrics(y_test_cnn, y_pred_cnn)

    # Transformer
    all_y_test, all_y_pred = [], []
    with torch.no_grad():
        for item in test_windows:
            conditions = torch.tensor(item['conditions'], dtype=torch.float32).unsqueeze(0)
            src_time = torch.tensor(item['time_series'], dtype=torch.float32).unsqueeze(0)
            y_test = item['vth_series']
            decoder_input = torch.zeros(1, 1, 1)
            for _ in range(src_time.size(1)):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.size(1))
                output = trans_model(src_time[:, :decoder_input.size(1), :], decoder_input, conditions, tgt_mask=tgt_mask)
                next_val = output[:, -1, :].unsqueeze(1)
                decoder_input = torch.cat([decoder_input, next_val], dim=1)
            all_y_test.extend(y_test.flatten())
            all_y_pred.extend(decoder_input[:, 1:, :].numpy().flatten())
    y_test_trans = np.array(all_y_test).reshape(-1, 1)
    y_pred_trans = np.array(all_y_pred).reshape(-1, 1)
    metrics_trans = calculate_metrics(y_test_trans, y_pred_trans)

    plot_single_accuracy(test_data_point[2], y_pred_ann, metrics_ann, 'ANN', 'exp1_accuracy_ANN.png', 0.0, cfg_output)
    plot_single_accuracy(test_data_point[2], y_pred_appd, metrics_appd, 'APPD-Net', 'exp1_accuracy_APPD-Net.png', 0.0, cfg_output)
    plot_single_accuracy(y_test_cnn, y_pred_cnn, metrics_cnn, '1D-CNN', 'exp1_accuracy_CNN.png', 0.0, cfg_output)
    plot_single_accuracy(y_test_trans, y_pred_trans, metrics_trans, 'Transformer', 'exp1_accuracy_Transformer.png', 0.0, cfg_output)

    accuracy_df = pd.DataFrame({
        'Model': ['ANN', 'APPD-Net', '1D-CNN', 'Transformer'],
        'RMSE(mV)': [metrics_ann['rmse'], metrics_appd['rmse'], metrics_cnn['rmse'], metrics_trans['rmse']],
        'MSE': [metrics_ann['mse'], metrics_appd['mse'], metrics_cnn['mse'], metrics_trans['mse']],
        'MAPE(%)': [metrics_ann['mape'], metrics_appd['mape'], metrics_cnn['mape'], metrics_trans['mape']],
        'R2': [metrics_ann['r2'], metrics_appd['r2'], metrics_cnn['r2'], metrics_trans['r2']],
    })


# =============================================================================
# Experiment 5
# =============================================================================
def build_point_ensemble(model_cls, bundle_path: Path):
    pack = torch.load(bundle_path, map_location='cpu')
    members = []
    for item in pack['members']:
        model = model_cls()
        model.load_state_dict(item['state_dict'])
        model.eval()
        scaler = dict_to_scaler(item.get('scaler'))
        members.append((model, scaler))
    return pack['meta'], members


def build_cnn_ensemble(bundle_path: Path):
    pack = torch.load(bundle_path, map_location='cpu')
    seq_len = int(pack['meta']['window_size_extrap'])
    members = []
    for item in pack['members']:
        model = CNN1D(seq_len=seq_len)
        model.load_state_dict(item['state_dict'])
        model.eval()
        scaler = dict_to_scaler(item.get('scaler'))
        members.append((model, scaler))
    return pack['meta'], members


def build_transformer_ensemble(bundle_path: Path):
    pack = torch.load(bundle_path, map_location='cpu')
    members = []
    for item in pack['members']:
        model = TimeSeriesTransformer()
        model.load_state_dict(item['state_dict'])
        model.eval()
        members.append(model)
    return pack['meta'], members


def run_exp5_from_weights(full_df: pd.DataFrame, weights_dir: Path, cfg_output: Dict[str, Any]):
    exp_dir = weights_dir / 'exp5'
    meta_ann, ann_ens = build_point_ensemble(ANN, exp_dir / 'ann_ensemble.pt')
    _, appd_ens = build_point_ensemble(APPD_Net, exp_dir / 'appd_ensemble.pt')
    _, cnn_ens = build_cnn_ensemble(exp_dir / 'cnn_ensemble.pt')
    _, trans_ens = build_transformer_ensemble(exp_dir / 'transformer_ensemble.pt')

    boundary = float(meta_ann['boundary_time'])
    ci_alpha = float(meta_ann['ci_alpha'])
    extrap_window_size = int(meta_ann['window_size_extrap'])
    boundary_atol = 1e-9
    zoom_low, zoom_high = 0.90, 1.10
    eps_plot = 1e-12


    conds_df = full_df[['Temperature', 'Vgs']].drop_duplicates().sort_values(by=['Vgs', 'Temperature']).reset_index(drop=True)
    for _, row in conds_df.iterrows():
        temp, vgs = float(row['Temperature']), float(row['Vgs'])
        cond_df = full_df[(full_df['Vgs'] == vgs) & (full_df['Temperature'] == temp)].sort_values('Time').reset_index(drop=True)
        X_pred_raw = cond_df[['Temperature', 'Vgs']].values.astype(float)
        t_pred = cond_df[['Time']].values.astype(float).reshape(-1)
        y_true = cond_df[['dVth']].values.astype(float).reshape(-1)
        N = len(cond_df)

        ann_preds = []
        for (m_cpu, sc) in ann_ens:
            X_full = sc.transform(np.hstack([X_pred_raw, t_pred.reshape(-1, 1)]))
            with torch.no_grad():
                ann_preds.append(m_cpu(torch.tensor(X_full, dtype=torch.float32)).numpy().reshape(-1))
        ann_preds = np.vstack(ann_preds) if ann_preds else np.full((1, N), np.nan)
        ann_mean, ann_low, ann_high = ensemble_mean_ci(ann_preds, ci_alpha)

        appd_preds = []
        for (m_cpu, sc) in appd_ens:
            X_scaled = sc.transform(X_pred_raw)
            with torch.no_grad():
                appd_preds.append(m_cpu.predict_dvth(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(t_pred.reshape(-1, 1), dtype=torch.float32)).numpy().reshape(-1))
        appd_preds = np.vstack(appd_preds) if appd_preds else np.full((1, N), np.nan)
        appd_mean, appd_low, appd_high = ensemble_mean_ci(appd_preds, ci_alpha)

        cnn_preds = []
        for (m_cpu, sc) in cnn_ens:
            cnn_preds.append(predict_cnn_curve(m_cpu, sc, cond_df, extrap_window_size).reshape(-1))
        cnn_preds = np.vstack(cnn_preds) if cnn_preds else np.full((1, N), np.nan)
        cnn_mean, cnn_low, cnn_high = ensemble_mean_ci(cnn_preds, ci_alpha)

        trans_preds = []
        for m_cpu in trans_ens:
            trans_preds.append(predict_transformer_curve(m_cpu, cond_df, boundary).reshape(-1))
        trans_preds = np.vstack(trans_preds) if trans_preds else np.full((1, N), np.nan)
        trans_mean, trans_low, trans_high = ensemble_mean_ci(trans_preds, ci_alpha)

        series_dict = {
            'True': y_true,
            'ANN_Mean': ann_mean, 'ANN_Low': ann_low, 'ANN_High': ann_high,
            'APPD_Mean': appd_mean, 'APPD_Low': appd_low, 'APPD_High': appd_high,
            'CNN_Mean': cnn_mean, 'CNN_Low': cnn_low, 'CNN_High': cnn_high,
            'TRANS_Mean': trans_mean, 'TRANS_Low': trans_low, 'TRANS_High': trans_high,
        }
        t_aug, series_aug, is_boundary_point = ensure_boundary_point(t_pred, series_dict, boundary, atol=boundary_atol)
        y_true_aug = series_aug['True']
        ann_mean_aug, ann_low_aug, ann_high_aug = series_aug['ANN_Mean'], series_aug['ANN_Low'], series_aug['ANN_High']
        appd_mean_aug, appd_low_aug, appd_high_aug = series_aug['APPD_Mean'], series_aug['APPD_Low'], series_aug['APPD_High']
        cnn_mean_aug, cnn_low_aug, cnn_high_aug = series_aug['CNN_Mean'], series_aug['CNN_Low'], series_aug['CNN_High']
        trans_mean_aug, trans_low_aug, trans_high_aug = series_aug['TRANS_Mean'], series_aug['TRANS_Low'], series_aug['TRANS_High']
        extrap_mask_aug = (t_aug >= boundary).astype(int)

        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(18, 7))
        t_plot = t_aug
        clip_pos = lambda arr: np.clip(np.asarray(arr, dtype=float), eps_plot, None)
        y_true_plot = clip_pos(y_true_aug)
        ann_plot = clip_pos(ann_mean_aug)
        appd_plot = clip_pos(appd_mean_aug)
        cnn_plot = clip_pos(cnn_mean_aug)
        trans_plot = clip_pos(trans_mean_aug)
        extrap_bool = (t_aug >= boundary)

        def masked_band(arr):
            arr = clip_pos(arr)
            return np.where(extrap_bool, arr, np.nan)

        ann_low_shade, ann_high_shade = masked_band(ann_low_aug), masked_band(ann_high_aug)
        appd_low_shade, appd_high_shade = masked_band(appd_low_aug), masked_band(appd_high_aug)
        cnn_low_shade, cnn_high_shade = masked_band(cnn_low_aug), masked_band(cnn_high_aug)
        trans_low_shade, trans_high_shade = masked_band(trans_low_aug), masked_band(trans_high_aug)

        def draw_one_axis(ax, do_zoom=False):
            ax.plot(t_plot, y_true_plot, 'ko-', label='True Data (Ground Truth)')
            ax.plot(t_plot, ann_plot, color='blue', linestyle=':', marker='s', markersize=4, label='ANN (Ens Mean)')
            ax.plot(t_plot, appd_plot, color='red', linestyle='--', marker='^', markersize=4, label='APPD-Net (Ens Mean)')
            ax.plot(t_plot, cnn_plot, color='green', linestyle='-.', marker='o', markersize=4, label='1D-CNN (Ens Mean)')
            ax.plot(t_plot, trans_plot, color='magenta', linestyle='--', marker='p', markersize=4, label='Transformer (Ens Mean)')
            ax.fill_between(t_plot, ann_low_shade, ann_high_shade, alpha=0.12, color='blue', label='ANN 95% Band (Extrap)')
            ax.fill_between(t_plot, appd_low_shade, appd_high_shade, alpha=0.12, color='red', label='APPD 95% Band (Extrap)')
            ax.fill_between(t_plot, cnn_low_shade, cnn_high_shade, alpha=0.12, color='green', label='CNN 95% Band (Extrap)')
            ax.fill_between(t_plot, trans_low_shade, trans_high_shade, alpha=0.12, color='magenta', label='Trans 95% Band (Extrap)')
            ax.axvline(x=boundary, color='grey', linestyle='--', label='Training Boundary')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Stress Time (s) [log scale]')
            if not do_zoom:
                ax.set_ylabel('dVth (mV) [log scale]')
            if do_zoom:
                ax.set_xlim(boundary, np.nanmax(t_plot) * 1.02)
                zoom_ylim = compute_zoom_ylim(t_aug, [y_true_aug, ann_mean_aug, ann_low_aug, ann_high_aug, appd_mean_aug, appd_low_aug, appd_high_aug, cnn_mean_aug, cnn_low_aug, cnn_high_aug, trans_mean_aug, trans_low_aug, trans_high_aug], boundary, low_margin=zoom_low, high_margin=zoom_high)
                if zoom_ylim is not None:
                    ax.set_ylim(zoom_ylim[0], zoom_ylim[1])

        draw_one_axis(ax_full, do_zoom=False)
        draw_one_axis(ax_zoom, do_zoom=True)
        ax_full.set_title(f'Time Extrapolation (Full): Vgs={vgs} V, T={temp} degC')
        ax_zoom.set_title(f'Zoom: Extrapolation Region (t >= {int(boundary)} s)')
        handles, labels = ax_zoom.get_legend_handles_labels()
        ax_zoom.legend(handles, labels, loc='best')
        plt.tight_layout()
        maybe_save_show(fig, f'exp5_T{temp}_V{vgs}.png'.replace('.', '_'), cfg_output)

        extrap_df = pd.DataFrame({
            'Time': t_aug,
            'Is_Extrapolation': extrap_mask_aug,
            'Is_BoundaryPoint': is_boundary_point,
            'True_dVth': y_true_aug,
            'ANN_Mean': ann_mean_aug, 'ANN_CI_Low': ann_low_aug, 'ANN_CI_High': ann_high_aug,
            'APPD_Mean': appd_mean_aug, 'APPD_CI_Low': appd_low_aug, 'APPD_CI_High': appd_high_aug,
            'CNN_Mean': cnn_mean_aug, 'CNN_CI_Low': cnn_low_aug, 'CNN_CI_High': cnn_high_aug,
            'Transformer_Mean': trans_mean_aug, 'Transformer_CI_Low': trans_low_aug, 'Transformer_CI_High': trans_high_aug,
        })


# =============================================================================
# Experiment 6
# =============================================================================
def run_exp6_from_weights(full_df: pd.DataFrame, weights_dir: Path, cfg_output: Dict[str, Any]):
    exp_dir = weights_dir / 'exp6'
    model, scaler, bundle = load_appd(exp_dir / 'appd_interpretability.pt')
    metrics_appd_80 = bundle['metrics']

    subset_150_sparse = full_df[full_df['Temperature'] == 150].sort_values('Vgs').drop_duplicates(subset=['Vgs'])
    X_sparse_vgs = scaler.transform(subset_150_sparse[['Temperature', 'Vgs']].values)
    with torch.no_grad():
        A_sparse_vgs, n_sparse_vgs, attn_sparse_vgs = model(torch.tensor(X_sparse_vgs, dtype=torch.float32))

    vgs_original_points = subset_150_sparse['Vgs'].unique()
    vgs_interp_points = np.linspace(vgs_original_points.min(), vgs_original_points.max(), 50)
    vgs_dense = np.unique(np.sort(np.concatenate([vgs_original_points, vgs_interp_points])))
    temp_fixed_vgs = np.full_like(vgs_dense, 150.0)
    X_dense_vgs_raw = np.vstack([temp_fixed_vgs, vgs_dense]).T
    X_dense_vgs_scaled = scaler.transform(X_dense_vgs_raw)
    with torch.no_grad():
        A_dense_vgs, n_dense_vgs, attn_dense_vgs = model(torch.tensor(X_dense_vgs_scaled, dtype=torch.float32))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(vgs_dense, A_dense_vgs.numpy(), 'b-', label='APPD-Net Interpolation')
    ax.plot(subset_150_sparse['Vgs'], A_sparse_vgs.numpy(), 'bo', mfc='none', markersize=8, label='Predictions at Original Points')
    ax.set_xlabel('Vgs (V)')
    ax.set_ylabel('Predicted Amplitude Factor A')
    ax.set_title('Predicted A vs. Vgs (at T=150 degC)')
    ax.grid(True)
    ax.set_yscale('log')
    ax.legend()
    maybe_save_show(fig, 'exp6_A_vs_vgs.png', cfg_output)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(vgs_dense, n_dense_vgs.numpy(), 'g-', label='APPD-Net Interpolation')
    ax.plot(subset_150_sparse['Vgs'], n_sparse_vgs.numpy(), 'gs', mfc='none', markersize=8, label='Predictions at Original Points')
    ax.set_xlabel('Vgs (V)')
    ax.set_ylabel('Predicted Power-Law Exponent n')
    ax.set_title('Predicted n vs. Vgs (at T=150 degC)')
    ax.set_ylim(0.15, 0.35)
    ax.grid(True)
    ax.legend()
    maybe_save_show(fig, 'exp6_n_vs_vgs.png', cfg_output)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(vgs_dense, attn_dense_vgs.numpy()[:, 0], 'r-', label='Attention to Temperature')
    ax.plot(vgs_dense, attn_dense_vgs.numpy()[:, 1], 'c-', label='Attention to Vgs')
    ax.plot(subset_150_sparse['Vgs'], attn_sparse_vgs.numpy()[:, 0], 'r^', mfc='none', markersize=8)
    ax.plot(subset_150_sparse['Vgs'], attn_sparse_vgs.numpy()[:, 1], 'cv', mfc='none', markersize=8)
    ax.set_xlabel('Vgs (V)')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Attention Weights vs. Vgs (at T=150 degC)')
    ax.legend()
    ax.grid(True)
    maybe_save_show(fig, 'exp6_attention_vs_vgs.png', cfg_output)

    subset_vgs4_sparse = full_df[full_df['Vgs'] == 4.0].sort_values('Temperature').drop_duplicates(subset=['Temperature'])
    X_sparse_temp = scaler.transform(subset_vgs4_sparse[['Temperature', 'Vgs']].values)
    with torch.no_grad():
        A_sparse_temp, n_sparse_temp, attn_sparse_temp = model(torch.tensor(X_sparse_temp, dtype=torch.float32))

    temp_original_points = subset_vgs4_sparse['Temperature'].unique()
    temp_interp_points = np.linspace(temp_original_points.min(), temp_original_points.max(), 50)
    temp_dense = np.unique(np.sort(np.concatenate([temp_original_points, temp_interp_points])))
    vgs_fixed_temp = np.full_like(temp_dense, 4.0)
    X_dense_temp_raw = np.vstack([temp_dense, vgs_fixed_temp]).T
    X_dense_temp_scaled = scaler.transform(X_dense_temp_raw)
    with torch.no_grad():
        A_dense_temp, n_dense_temp, attn_dense_temp = model(torch.tensor(X_dense_temp_scaled, dtype=torch.float32))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(temp_dense, A_dense_temp.numpy(), 'm-', label='APPD-Net Interpolation')
    ax.plot(subset_vgs4_sparse['Temperature'], A_sparse_temp.numpy(), 'mo', mfc='none', markersize=8, label='Predictions at Original Points')
    ax.set_xlabel('Temperature (degC)')
    ax.set_ylabel('Predicted Amplitude Factor A')
    ax.set_title('Predicted A vs. Temperature (at Vgs=4V)')
    ax.grid(True)
    ax.set_yscale('log')
    ax.legend()
    maybe_save_show(fig, 'exp6_A_vs_temp.png', cfg_output)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(temp_dense, n_dense_temp.numpy(), 'y-', label='APPD-Net Interpolation')
    ax.plot(subset_vgs4_sparse['Temperature'], n_sparse_temp.numpy(), 'ys', mfc='none', markersize=8, label='Predictions at Original Points')
    ax.set_xlabel('Temperature (degC)')
    ax.set_ylabel('Predicted Power-Law Exponent n')
    ax.set_title('Predicted n vs. Temperature (at Vgs=4V)')
    ax.set_ylim(0.15, 0.35)
    ax.grid(True)
    ax.legend()
    maybe_save_show(fig, 'exp6_n_vs_temp.png', cfg_output)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(temp_dense, attn_dense_temp.numpy()[:, 0], 'r-', label='Attention to Temperature')
    ax.plot(temp_dense, attn_dense_temp.numpy()[:, 1], 'c-', label='Attention to Vgs')
    ax.plot(subset_vgs4_sparse['Temperature'], attn_sparse_temp.numpy()[:, 0], 'r^', mfc='none', markersize=8)
    ax.plot(subset_vgs4_sparse['Temperature'], attn_sparse_temp.numpy()[:, 1], 'cv', mfc='none', markersize=8)
    ax.set_xlabel('Temperature (degC)')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Attention Weights vs. Temperature (at Vgs=4V)')
    ax.legend()
    ax.grid(True)
    maybe_save_show(fig, 'exp6_attention_vs_temp.png', cfg_output)

    temp_min, temp_max = full_df['Temperature'].min(), full_df['Temperature'].max()
    vgs_min, vgs_max = full_df['Vgs'].min(), full_df['Vgs'].max()
    temp_grid = np.linspace(temp_min, temp_max, 100)
    vgs_grid = np.linspace(vgs_min, vgs_max, 100)
    T_grid, VGS_grid = np.meshgrid(temp_grid, vgs_grid)
    grid_points_raw = np.vstack([T_grid.ravel(), VGS_grid.ravel()]).T
    grid_points_scaled = scaler.transform(grid_points_raw)
    with torch.no_grad():
        A_grid_pred, n_grid_pred, _ = model(torch.tensor(grid_points_scaled, dtype=torch.float32))
    A_grid = A_grid_pred.numpy().reshape(T_grid.shape)
    n_grid = n_grid_pred.numpy().reshape(T_grid.shape)

    fig_A, ax_A = plt.subplots(figsize=(10, 8))
    levels_A = np.logspace(np.log10(np.maximum(A_grid.min(), 1e-12)), np.log10(A_grid.max()), 15)
    contour_A = ax_A.contourf(VGS_grid, T_grid, A_grid, levels=levels_A, cmap='viridis', norm=plt.cm.colors.LogNorm())
    cbar_A = fig_A.colorbar(contour_A)
    cbar_A.set_label('Amplitude Factor A [log scale]')
    ax_A.set_xlabel('Vgs (V)')
    ax_A.set_ylabel('Temperature (degC)')
    ax_A.set_title('Contour Plot of Predicted Amplitude Factor A')
    ax_A.scatter(full_df['Vgs'], full_df['Temperature'], c='red', s=10, label='Experimental Points')
    ax_A.legend()
    maybe_save_show(fig_A, 'exp6_contour_A.png', cfg_output)

    fig_n, ax_n = plt.subplots(figsize=(10, 8))
    levels_n = np.linspace(n_grid.min(), n_grid.max(), 15)
    contour_n = ax_n.contourf(VGS_grid, T_grid, n_grid, levels=levels_n, cmap='cividis')
    cbar_n = fig_n.colorbar(contour_n)
    cbar_n.set_label('Power-Law Exponent n')
    ax_n.set_xlabel('Vgs (V)')
    ax_n.set_ylabel('Temperature (degC)')
    ax_n.set_title('Contour Plot of Predicted Power-Law Exponent n')
    ax_n.scatter(full_df['Vgs'], full_df['Temperature'], c='red', s=10, label='Experimental Points')
    ax_n.legend()
    maybe_save_show(fig_n, 'exp6_contour_n.png', cfg_output)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Load pretrained weights for Experiments 1, 5, and 6 and generate plots directly.')
    parser.add_argument('--data-dir', type=str, default=None, help='Data directory. Default: same directory as this script.')
    parser.add_argument('--weights-dir', type=str, default='weights_exp156', help='Weights directory.')
    parser.add_argument('--figure-dir', type=str, default='figures_exp156_pretrained', help='Figure output directory.')
    
    parser.add_argument('--config-json', type=str, default=None, help='Optional custom configuration JSON.')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    cfg = default_config(script_dir)
    if args.data_dir:
        cfg['data']['base_path'] = args.data_dir
    cfg['output']['figure_dir'] = args.figure_dir
    if args.config_json:
        with open(args.config_json, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)
        for k, v in user_cfg.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v

    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        raise FileNotFoundError(f'Weights directory does not exist: {weights_dir}')

    ensure_dir(Path(cfg['output']['figure_dir']))
    full_df = load_and_prepare_data(cfg)

    run_exp1_from_weights(full_df, weights_dir, cfg['output'])
    run_exp5_from_weights(full_df, weights_dir, cfg['output'])
    run_exp6_from_weights(full_df, weights_dir, cfg['output'])

    if cfg['output']['show_fig']:
        plt.show()
        plt.close('all')

    print(f"Done: loaded weights from {weights_dir.resolve()} and saved figures to {Path(cfg['output']['figure_dir']).resolve()}. Figures were also displayed on screen.")


if __name__ == '__main__':
    main()
