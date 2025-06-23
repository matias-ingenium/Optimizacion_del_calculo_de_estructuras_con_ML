import torch                     # PyTorch: librería principal
import torch.nn as nn            # Para definir la red neuronal
import torch.optim as optim      # Para los optimizadores (SGD, Adam, etc.)
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

class RED_NEURONAL:
    def __init__(self,
                 n_entrada,
                 m_salida,
                 training_set,
                 salidas,
                 capas_ocultas=2,
                 neuronas_por_capa=32,
                 funcion_costo= None,
                 lr=0.002,
                 tol=1e-7,
                 no_improvement_limit=200,
                 min_delta = 1e-7,
                 max_epochs=500):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Usando dispositivo:", self.device)

        self.n = n_entrada
        self.m = m_salida
        self.capas = capas_ocultas
        self.neuronas = neuronas_por_capa
        self.lr = lr
        self.max_epochs = max_epochs
        self.tol = tol
        self.no_improvement_limit = no_improvement_limit
        self.min_delta = min_delta

        # Preparación de datos
        self.X = torch.tensor(training_set, dtype=torch.float32).to(self.device)
        Y_np = np.stack(salidas)
        self.Y = torch.tensor(Y_np, dtype=torch.float32).to(self.device)

        print("Forma de X:", self.X.shape)
        print("Forma de Y:", self.Y.shape)

        # Red neuronal
        self.model = self._construir_modelo().to(self.device)

        # Función de costo
        if funcion_costo is None:
            self.loss_fn = lambda y_pred, y_true: torch.mean((y_pred - y_true) ** 2)  # MSE por defecto
        else:
            self.loss_fn = funcion_costo


        # Optimizador
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _construir_modelo(self):
        capas = []
        capas.append(nn.Linear(self.n, self.neuronas))
        capas.append(nn.PReLU())
        for _ in range(self.capas - 1):
            capas.append(nn.Linear(self.neuronas, self.neuronas))
            capas.append(nn.PReLU())
        capas.append(nn.Linear(self.neuronas, self.m))
        return nn.Sequential(*capas)

    def entrenar(self):
        best_loss = float("inf")
        best_model_state = None
        epochs_no_improve = 0
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=250)
        for epoch in range(self.max_epochs):
            self.model.train()
            y_pred = self.model(self.X)
            loss = self.loss_fn(y_pred, self.Y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            scheduler.step(loss.item())
            # Guardar el mejor modelo
            if loss.item() < best_loss - self.min_delta:
                best_loss = loss.item()
                best_model_state = self.model.state_dict()  # Guardar estado
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Condición de parada temprana
            if loss.item() < self.tol or epochs_no_improve >= self.no_improvement_limit:
                print(f"Parada temprana en la época {epoch}, pérdida: {loss.item():.10f}")
                break

            if epoch % 200 == 0:
                print(f"Época {epoch}, Pérdida: {loss.item():.10f}, lr: {scheduler.get_last_lr()[0]}")

        # Restaurar el mejor modelo
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Modelo restaurado a la mejor pérdida: {best_loss:.10f}")


    def predecir(self, entrada):
        self.model.eval()
        entrada_tensor = torch.tensor(entrada, dtype=torch.float32).to(self.device)
        if entrada_tensor.ndim == 1:
            entrada_tensor = entrada_tensor.unsqueeze(0)
        with torch.no_grad():
            prediccion = self.model(entrada_tensor)
        return prediccion.cpu().numpy()
