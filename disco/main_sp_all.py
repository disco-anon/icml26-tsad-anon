import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.font_manager as fm

# 设置中文字体，解决中文乱码问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# -------------------------- 1. 数据生成与预处理 --------------------------
def generate_time_series(normal_samples=1000, anomaly_samples=200, seq_len=20, n_features=1):
    """生成含简单/复杂正常样本+异常样本的时间序列数据"""
    # 正常数据：正弦波+噪声（含10%“复杂正常样本”）
    t = np.linspace(0, normal_samples * 0.1, normal_samples * seq_len)
    normal_data = np.sin(t) + 0.1 * np.random.randn(normal_samples * seq_len)

    # 加入“复杂正常样本”（偏离较小，仍属正常，但重构难度高）
    difficult_idx = np.random.choice(normal_samples * seq_len, size=int(normal_samples * seq_len * 0.1))
    normal_data[difficult_idx] += 0.6 * np.random.randn(len(difficult_idx))  # 更大噪声模拟复杂样本

    normal_data = normal_data.reshape(normal_samples, seq_len, n_features)

    # 异常数据：完全随机分布（与正常模式差异大）
    anomaly_data = np.random.uniform(low=-3, high=3, size=(anomaly_samples * seq_len, n_features))
    anomaly_data = anomaly_data.reshape(anomaly_samples, seq_len, n_features)

    # 标签：0=正常，1=异常
    y = np.concatenate([np.zeros(normal_samples), np.ones(anomaly_samples)])
    X = np.concatenate([normal_data, anomaly_data], axis=0)

    # 打乱数据
    shuffle_idx = np.random.permutation(len(X))
    X, y = X[shuffle_idx], y[shuffle_idx]
    return X, y

class TimeSeriesDataset(Dataset):
    """自定义数据集：返回样本+索引（用于映射样本权重）"""

    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx  # (样本, 样本索引)


# -------------------------- 2. LSTM-VAE模型定义 --------------------------
class LSTMVAE(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, latent_dim=32, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 编码器：LSTM + 均值/方差层
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # 解码器：输入映射 + LSTM + 输出层
        self.decoder_input = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_output = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        """编码：输入序列 → 潜在变量均值/方差"""
        # x: (batch_size, seq_len, input_size)
        _, (hidden, _) = self.encoder_lstm(x)  # hidden: (num_layers, batch_size, hidden_size)
        hidden = hidden[-1]  # 取最后一层隐藏态：(batch_size, hidden_size)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        """重参数化：从N(mu, var)采样潜在变量"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        """解码：潜在变量 → 重构序列"""
        # z: (batch_size, latent_dim)
        batch_size = z.size(0)

        # 初始化解码器隐藏态
        hidden = self.decoder_input(z).unsqueeze(0)  # (1, batch_size, hidden_size)
        hidden = hidden.repeat(self.num_layers, 1, 1)  # (num_layers, batch_size, hidden_size)
        cell = torch.zeros_like(hidden)

        # 逐步生成序列（自回归）
        recon_seq = []
        decoder_input = torch.zeros(batch_size, 1, self.hidden_size, device=z.device)
        for _ in range(seq_len):
            out, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            recon_seq.append(out)
            decoder_input = out  # 下一时间步输入=当前输出

        # 合并序列并映射到原始维度
        recon_seq = torch.cat(recon_seq, dim=1)  # (batch_size, seq_len, hidden_size)
        return self.fc_output(recon_seq)  # (batch_size, seq_len, input_size)

    def forward(self, x):
        """前向传播：编码→采样→解码"""
        seq_len = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, seq_len)
        return recon_x, mu, logvar


# -------------------------- 3. 自步学习核心工具函数 --------------------------
def compute_sample_errors(model, dataloader, device):
    """计算每个样本的重构误差（MSE），用于自步学习筛选样本"""
    model.eval()
    sample_errors = np.zeros(len(dataloader.dataset))  # 按样本索引存储误差
    with torch.no_grad():
        for batch, indices in dataloader:
            batch = batch.to(device)
            recon_batch, _, _ = model(batch)
            # 计算每个样本的MSE（序列长度+特征维度平均）
            mse = nn.MSELoss(reduction="none")(recon_batch, batch).mean(dim=(1, 2))
            # 映射到全局样本索引
            for i, idx in enumerate(indices):
                sample_errors[idx] = mse[i].cpu().numpy()
    return sample_errors


def get_sample_weights(sample_errors, lambda_val):
    """根据当前lambda值生成样本权重：误差≤lambda→权重1，否则→权重0"""
    return (sample_errors <= lambda_val).astype(float)


def weighted_vae_loss(recon_x, x, mu, logvar, sample_weights):
    """带样本权重的VAE损失（重构损失+KL散度均加权）"""
    # 重构损失：按样本权重加权
    recon_loss = nn.MSELoss(reduction="none")(recon_x, x).mean(dim=(1, 2))  # (batch_size,)
    recon_loss = torch.sum(recon_loss * sample_weights)  # 加权求和

    # KL散度：按样本权重加权
    kl_loss = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()) * sample_weights.unsqueeze(1))
    return recon_loss + kl_loss


def train_lstm_vae_full(model, dataloader, sample_weights, epochs, lr, device, stage_name):
    """完整的LSTM-VAE训练（用于自步学习的每个阶段）"""
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 每个阶段独立优化器
    model.train()
    stage_loss_history = []

    for epoch in range(epochs):
        total_loss = 0.0
        for batch, indices in dataloader:
            batch = batch.to(device)
            # 获取当前批次样本的权重
            batch_weights = torch.tensor(
                [sample_weights[idx] for idx in indices],
                dtype=torch.float32,
                device=device
            )

            # 前向传播+损失计算+反向传播
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = weighted_vae_loss(recon_batch, batch, mu, logvar, batch_weights)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 记录平均损失（按样本数归一化）
        avg_loss = total_loss / len(dataloader.dataset)
        stage_loss_history.append(avg_loss)

        # 每10轮打印进度
        if (epoch + 1) % 10 == 0:
            print(f"[{stage_name}] 轮次 {epoch + 1:2d}/{epochs} | 平均损失: {avg_loss:.4f}")

    return model, stage_loss_history


# -------------------------- 4. 两阶段训练主流程 --------------------------
def main():
    # -------------------------- 配置参数 --------------------------
    # 数据参数
    seq_len = 20  # 时间序列长度
    n_features = 1  # 特征数（单变量）
    normal_samples = 1000  # 正常样本数
    anomaly_samples = 200  # 异常样本数

    # 训练参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    # 第一阶段：常规训练（无自步学习）
    normal_train_epochs = 80  # 常规训练迭代次数
    normal_lr = 1e-3  # 常规训练学习率

    # 第二阶段：自步学习（多阶段，每个阶段跑完整LSTM训练）
    spl_stages = 5  # 自步学习总阶段数
    spl_train_epochs = 50  # 每个自步阶段的LSTM迭代次数
    spl_lr = 5e-4  # 自步学习阶段学习率（略小，避免震荡）
    lambda_quantile_range = (30, 90)  # lambda分位数范围（从30%→90%）

    print(f"使用设备: {device}")
    print(f"自步学习配置：{spl_stages}个阶段，每个阶段训练{spl_train_epochs}轮\n")

    # -------------------------- 步骤1：数据准备 --------------------------
    # 生成数据
    X, y = generate_time_series(
        normal_samples=normal_samples,
        anomaly_samples=anomaly_samples,
        seq_len=seq_len,
        n_features=n_features
    )

    # 归一化（MinMax→[0,1]）
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, n_features)).reshape(X.shape)

    # 划分训练集（仅正常样本）和测试集（正常+异常）
    train_mask = y == 0  # 训练集只含正常样本
    X_train = X_scaled[train_mask][:800]  # 800个正常样本用于训练
    X_test = np.concatenate([
        X_scaled[~train_mask],  # 所有异常样本
        X_scaled[train_mask][800:]  # 200个正常样本（用于测试）
    ])
    y_test = np.concatenate([
        np.ones(len(X_scaled[~train_mask])),  # 异常标签
        np.zeros(200)  # 正常标签
    ])

    # 创建数据集和加载器
    train_dataset = TimeSeriesDataset(X_train)
    test_dataset = TimeSeriesDataset(X_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------- 步骤2：第一阶段：常规训练 --------------------------
    print("=" * 50)
    print("第一阶段：常规训练（无自步学习）")
    print("=" * 50)

    # 初始化模型
    base_model = LSTMVAE(
        input_size=n_features,
        hidden_size=64,
        latent_dim=32,
        num_layers=2
    ).to(device)

    # 常规训练（无权重，所有样本参与）
    base_model, normal_loss_history = train_lstm_vae_full(
        model=base_model,
        dataloader=train_loader,
        sample_weights=np.ones(len(X_train)),  # 所有样本权重=1
        epochs=normal_train_epochs,
        lr=normal_lr,
        device=device,
        stage_name="常规训练"
    )

    # 常规训练后性能评估
    def evaluate_model(model, train_loader, test_loader, y_test, device, title):
        """评估模型异常检测性能"""
        # 计算训练集误差（用于确定阈值：均值+3倍标准差）
        train_errors = compute_sample_errors(model, train_loader, device)
        threshold = np.mean(train_errors) + 3 * np.std(train_errors)

        # 计算测试集误差并检测异常
        test_errors = compute_sample_errors(model, test_loader, device)
        y_pred = (test_errors > threshold).astype(int)

        # 计算评估指标
        TP = np.sum((y_pred == 1) & (y_test == 1))
        TN = np.sum((y_pred == 0) & (y_test == 0))
        FP = np.sum((y_pred == 1) & (y_test == 0))
        FN = np.sum((y_pred == 0) & (y_test == 1))

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n[{title}] 性能评估:")
        print(f"阈值: {threshold:.4f} | 准确率: {accuracy:.4f} | F1分数: {f1:.4f}")
        print(f"混淆矩阵: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
        return {"threshold": threshold, "f1": f1, "train_errors": train_errors, "test_errors": test_errors}

    # 评估常规训练结果
    normal_eval = evaluate_model(
        model=base_model,
        train_loader=train_loader,
        test_loader=test_loader,
        y_test=y_test,
        device=device,
        title="常规训练后"
    )

    # -------------------------- 步骤3：第二阶段：自步学习（多阶段完整训练） --------------------------
    print("\n" + "=" * 50)
    print("第二阶段：自步学习（每个阶段跑完整LSTM训练）")
    print("=" * 50)

    # 初始化自步学习状态
    current_model = base_model  # 初始模型=常规训练后的模型
    spl_records = {
        "stage_loss": [],  # 每个阶段的损失曲线
        "lambda_vals": [],  # 每个阶段的lambda值
        "included_ratio": [],  # 每个阶段参与训练的样本比例
        "stage_eval": []  # 每个阶段的性能评估
    }

    # 计算初始误差（用于确定第一个阶段的lambda）
    initial_train_errors = compute_sample_errors(current_model, train_loader, device)

    # 外层循环：自步学习阶段
    for stage in range(1, spl_stages + 1):
        print(f"\n=== 自步学习阶段 {stage}/{spl_stages} ===")

        # -------------------------- 3.1 计算当前阶段的lambda和样本权重 --------------------------
        # lambda分位数：从30%线性增加到90%（随阶段递进）
        quantile = lambda_quantile_range[0] + (lambda_quantile_range[1] - lambda_quantile_range[0]) * (stage - 1) / (
                    spl_stages - 1)
        lambda_val = np.percentile(initial_train_errors, quantile)  # 基于初始误差分布定lambda

        # 计算样本权重（误差≤lambda→1，否则→0）
        current_train_errors = compute_sample_errors(current_model, train_loader, device)
        sample_weights = get_sample_weights(current_train_errors, lambda_val)
        included_ratio = np.mean(sample_weights)  # 参与训练的样本比例

        print(f"当前lambda值: {lambda_val:.4f} | 参与训练样本比例: {included_ratio:.2%}")

        # -------------------------- 3.2 本阶段：完整LSTM-VAE训练 --------------------------
        current_model, stage_loss = train_lstm_vae_full(
            model=current_model,
            dataloader=train_loader,
            sample_weights=sample_weights,
            epochs=spl_train_epochs,
            lr=spl_lr,
            device=device,
            stage_name=f"自步阶段{stage}"
        )

        # -------------------------- 3.3 记录本阶段结果并评估 --------------------------
        spl_records["stage_loss"].append(stage_loss)
        spl_records["lambda_vals"].append(lambda_val)
        spl_records["included_ratio"].append(included_ratio)

        # 评估本阶段模型性能
        stage_eval = evaluate_model(
            model=current_model,
            train_loader=train_loader,
            test_loader=test_loader,
            y_test=y_test,
            device=device,
            title=f"自步阶段{stage}后"
        )
        spl_records["stage_eval"].append(stage_eval)

    # -------------------------- 步骤4：结果可视化 --------------------------
    print("\n" + "=" * 50)
    print("结果可视化")
    print("=" * 50)

    # 1. 训练损失对比（常规训练 + 各自步阶段）
    plt.figure(figsize=(15, 10))

    # 子图1：常规训练损失
    plt.subplot(2, 2, 1)
    plt.plot(normal_loss_history, label="常规训练", color="blue")
    plt.title("常规训练损失曲线")
    plt.xlabel("轮次")
    plt.ylabel("平均损失")
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图2：各自步阶段损失
    plt.subplot(2, 2, 2)
    for i, (loss, lambda_val) in enumerate(zip(spl_records["stage_loss"], spl_records["lambda_vals"])):
        plt.plot(loss, label=f"阶段{i + 1} (λ={lambda_val:.3f})")
    plt.title("各自步阶段损失曲线")
    plt.xlabel("轮次")
    plt.ylabel("平均损失")
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图3：lambda值变化
    plt.subplot(2, 2, 3)
    plt.plot(range(1, spl_stages + 1), spl_records["lambda_vals"], marker="o", linewidth=2, color="orange")
    plt.title("自步学习lambda值变化")
    plt.xlabel("自步阶段")
    plt.ylabel("lambda值")
    plt.xticks(range(1, spl_stages + 1))
    plt.grid(alpha=0.3)

    # 子图4：参与训练样本比例变化
    plt.subplot(2, 2, 4)
    plt.plot(range(1, spl_stages + 1), spl_records["included_ratio"], marker="s", linewidth=2, color="green")
    plt.title("参与训练样本比例变化")
    plt.xlabel("自步阶段")
    plt.ylabel("参与比例")
    plt.xticks(range(1, spl_stages + 1))
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 2. 最终模型：正常/异常样本重构示例
    final_model = current_model
    final_test_errors = spl_records["stage_eval"][-1]["test_errors"]
    final_threshold = spl_records["stage_eval"][-1]["threshold"]

    # 获取测试集中的正常/异常样本索引
    anomaly_test_idx = np.where(y_test == 1)[0]
    normal_test_idx = np.where(y_test == 0)[0]

    # 绘制5个异常样本重构
    plt.figure(figsize=(12, 10))
    for i in range(min(5, len(anomaly_test_idx))):
        idx = anomaly_test_idx[i]
        sample = torch.tensor(X_test[idx:idx + 1], dtype=torch.float32).to(device)

        final_model.eval()
        with torch.no_grad():
            recon_sample, _, _ = final_model(sample)

        # 绘图
        plt.subplot(5, 1, i + 1)
        plt.plot(sample[0].cpu().numpy(), label="原始序列", alpha=0.8, color="blue")
        plt.plot(recon_sample[0].cpu().numpy(), label="重构序列", alpha=0.8, color="red")
        plt.axhline(y=final_threshold, color="black", linestyle="--", label=f"阈值={final_threshold:.3f}")
        plt.title(f"异常样本{i + 1} | 重构误差: {final_test_errors[idx]:.3f}")
        plt.legend(loc="upper right")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 绘制5个正常样本重构
    plt.figure(figsize=(12, 10))
    for i in range(min(5, len(normal_test_idx))):
        idx = normal_test_idx[i]
        sample = torch.tensor(X_test[idx:idx + 1], dtype=torch.float32).to(device)

        final_model.eval()
        with torch.no_grad():
            recon_sample, _, _ = final_model(sample)

        # 绘图
        plt.subplot(5, 1, i + 1)
        plt.plot(sample[0].cpu().numpy(), label="原始序列", alpha=0.8, color="blue")
        plt.plot(recon_sample[0].cpu().numpy(), label="重构序列", alpha=0.8, color="green")
        plt.axhline(y=final_threshold, color="black", linestyle="--", label=f"阈值={final_threshold:.3f}")
        plt.title(f"正常样本{i + 1} | 重构误差: {final_test_errors[idx]:.3f}")
        plt.legend(loc="upper right")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()