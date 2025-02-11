import matplotlib.pyplot as plt
import numpy as np
import torch

from numpy.polynomial.legendre import leggauss
from Neural_Network import Neural_Network
from datetime import datetime

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)

## ----------------------Neural Network Parameters------------------##

deep_layers = 5
hidden_layers_dimension = 15

## ----------------------Training Parameters------------------##

batch_size = 20
epochs = 7500
learning_rate = 10e-5
optimizer = "Adam"  # Adam or SGD

NN = Neural_Network(input_dimension = 1,
                    output_dimension = 1,
                    deep_layers = deep_layers,
                    hidden_layers_dimension = hidden_layers_dimension,
                    optimizer = optimizer,
                    learning_rate = learning_rate)

integration_points = torch.linspace(0, 1, 1000, requires_grad = False).unsqueeze(-1)
estimator_points = torch.linspace(0, 1, 1000, requires_grad = False).unsqueeze(-1)

integration_nodes, integration_weights = leggauss(3)
integration_nodes = torch.tensor(integration_nodes).unsqueeze(1)
integration_weights = torch.tensor(integration_weights).unsqueeze(1)

def integrate(function, integration_points):
    
    elements_diameter = integration_points[1:] - integration_points[:-1]
    sum_integration_points = integration_points[1:] + integration_points[:-1]
    
    mapped_weights = (0.5 * elements_diameter * integration_weights.T).unsqueeze(-1)
    mapped_integration_nodes = 0.5 * elements_diameter * integration_nodes.T + 0.5 * sum_integration_points
    mapped_integration_nodes_single_dimension = mapped_integration_nodes.view(-1,1)
 
    function_values = function(mapped_integration_nodes_single_dimension) 
    
    nodes_value = function_values.view(elements_diameter.size(0), 
                                       mapped_weights.size(1),
                                       function_values.size(1)) 
 
    integral_value = torch.sum(mapped_weights.unsqueeze(-1) * nodes_value.unsqueeze(-1), dim = 1)
    
    return torch.sum(integral_value)

def integrate_pointwise(function, integration_points):

 
    function_values = function(integration_points) 
    
    return function_values.squeeze()

rhs = lambda x: np.pi**2 * torch.sin(np.pi * x) 
f = lambda x: (NN.laplacian(x) + rhs(x))**2
exact = lambda x: torch.sin(np.pi*x)

# beta = lambda t: np.exp(-0.0005 * t)

# beta = lambda t: 1 /(t + 1)**(0.5)

beta = lambda t: xd_list[t]

# beta = lambda t: 0

M_previous = 0.
M_cal_previous = 0.
M_estimator_previous = 0.

def compute_loss(t, M_previous, M_cal_previous, M_estimator_previous):

    # Compute Values for Loss

    loss_function = torch.nn.L1Loss(reduction = 'sum')

    x = torch.rand(batch_size)
    
    if (t == 0):
        M_previous = torch.mean(f(x))
        M_cal_previous = integrate(f, integration_points)
        M_estimator_previous = integrate_pointwise(f, estimator_points)

    # Compute Errors
    
    F = torch.mean(f(x))                    
    M = beta(t) * F + (1 - beta(t)) * M_previous
    
    # Compute Exact value for error 
    
    F_cal = integrate(f, integration_points)
    M_cal = beta(t) * F_cal + (1 - beta(t)) * M_cal_previous
    
    # Compute value for memory estimator 
    
    F_estimator = integrate_pointwise(f, estimator_points)
    M_estimator = beta(t) * F_estimator + (1 - beta(t)) * M_estimator_previous
    
    memory_estimator = torch.sum(torch.abs(F_estimator - M_estimator))
    
    # Compute Total error, 
    
    total_error = abs(F_cal.item() - M.item())
    memory_error = abs(F_cal.item() - M_cal.item())
    integration_error = abs(M_cal.item() - M.item())
    
    # Compute loss function value
    
    loss_value = loss_function(F, torch.zeros_like(F,requires_grad = True))

    return loss_value, M, F_cal, M_cal, M_estimator, total_error, memory_error, integration_error , memory_estimator


nb_betas = 100
betas = torch.linspace(0, 1.-1e-5, nb_betas).unsqueeze(-1)
M_estimators_previous = 0.

def plot_estimator(epoch, M_estimators_previous):
    
    if epoch == 0: 
        M_estimators_previous = torch.ones_like(betas) *  integrate_pointwise(f, estimator_points).unsqueeze(-1).T

    F_estimators = integrate_pointwise(f, estimator_points).unsqueeze(-1)
    
    M_estimators = betas * F_estimators.T + (1 - betas) * M_estimators_previous
    
    memory_estimators = torch.sum(torch.abs(F_estimators.T - M_estimators), dim = 1)

    return memory_estimators, M_estimators

loss_list = []
M_list = []
F_cal_list = []

total_error_list = []
sum_error_list = []

memory_error_list = []
integration_error_list = []
memory_estimator_list = []

xd_list = [0.]

# alpha_list = [None, None]

start_time = datetime.now()

plot_estimator_data = torch.zeros((epochs, nb_betas))

print(f"{'='*30} Training {'='*30}")
for epoch in range(epochs):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{epochs} {'='*20}")
        
    loss_value, M, F_cal, M_cal, M_estimator, total_error, memory_error, integration_error , memory_estimator = compute_loss(epoch, M_previous, M_cal_previous, M_estimator_previous)
    
    M_previous = M.item()
    M_cal_previous = M_cal.item()
    M_estimator_previous = M_estimator
    
    plot_estimator_data[epoch, :], M_estimators_previous = plot_estimator(epoch, M_estimators_previous) # WARNING!!! no deberia usar M_estimator_previous, eso genera el gap
    
    # if epoch >= 2:    
    #     alpha = np.log(integration_error/integration_error_list[-1])/np.log(epoch/ (epoch - 1.))
    #     alpha_list.append(alpha)
        
    loss_list.append(loss_value.item())
    M_list.append(M.item())
    F_cal_list.append(F_cal.item())
    
    total_error_list.append(total_error)
    sum_error_list.append(memory_error + integration_error)
    
    memory_error_list.append(memory_error)
    integration_error_list.append(integration_error)
    memory_estimator_list.append(memory_estimator.item())
    
    xd =  1 - (abs(memory_estimator.item() - memory_error_list[-1])+1)/(memory_estimator.item() + memory_error_list[-1]+1)
    xd_list.append(xd)
    
    print(f"Loss: {loss_value.item():.8f}")
    
    NN.optimizer_step(loss_value)
    
end_time = datetime.now()

execution_time = end_time - start_time

print(f"Training time: {execution_time}")

plot_points = torch.linspace(0, 1, 1000)

exact_evaluation = exact(plot_points)
exact_evaluation = exact_evaluation.cpu().detach().numpy()

NN_evaluation = NN(plot_points)
NN_evaluation = NN_evaluation.cpu().detach().numpy()

plot_points = plot_points.cpu().detach().numpy()

convergence = lambda x: 3.5 / np.sqrt(x + 1)

beta_values = convergence(np.linspace(0 , epochs - 1, epochs))

figure_solution, axis_solution = plt.subplots(dpi = 500)

axis_solution.plot(plot_points,
                   exact_evaluation,
                   label = r"$u^*$")

axis_solution.plot(plot_points,
                   NN_evaluation,
                   linestyle = ":",
                   label = r"$u_{NN}$")

axis_solution.legend()

axis_solution.set(title = "PINNs final solution", 
                  xlabel = "x", 
                  ylabel = "u(x)")

figure_loss, axis_loss = plt.subplots(dpi  = 500)

axis_loss.semilogy(loss_list,
                    linestyle = ":",
                    label = r"Loss $(F)$")

axis_loss.semilogy(M_list,
                    linestyle = "--",
                    label = r"Memory Loss $(M)$")

axis_loss.semilogy(F_cal_list,
                    linestyle = "-",
                    alpha = 0.8,
                    color = "black",
                    label = r"Exact Loss($\mathcal{F}$)")

axis_loss.legend()

axis_loss.set(title = "Loss evolution",
              xlabel = "# Epochs",
              ylabel = "Loss value")

figure_error, axis_error = plt.subplots(dpi = 500)

axis_error.semilogy(integration_error_list,
                    linestyle = ":",
                    alpha = 0.8,
                    label = r"Integration error $(|\mathcal{M} - M|)$")

axis_error.semilogy(memory_error_list,
                    linestyle = "--",
                    label = r"Memory error $(|\mathcal{F} - \mathcal{M}|)$")

axis_error.semilogy(memory_estimator_list,
                    linestyle = "-.",
                    label = r"Estimator $(|\bar{F} - \bar{M}|)$")

axis_error.semilogy(beta_values,
                    linestyle = "-",
                    alpha = 0.6,
                    color = "black",
                    label = "Theorical convergence")

# axis_error.plot(alpha_list,
#                 linestyle = ":",
#                 label = "Numerical convergence")

axis_error.set(title = "Error Comparison",
               xlabel = "# Epochs",
               ylabel = "Error value")

axis_error.legend()

plt.show()

fig_3d = plt.figure(figsize=(10,8))
ax_3d = fig_3d.add_subplot(111, 
                           projection = '3d')

epochs_range = torch.arange(1, epochs)  
betas_np = betas.squeeze().cpu().detach().numpy() 
epochs_np = epochs_range.cpu().detach().numpy()

B, E = np.meshgrid(betas_np, 
                   epochs_np)

plot_estimator_data_np = plot_estimator_data.cpu().detach().numpy()

plot_estimator_data_scaled = np.log10(plot_estimator_data_np[1:,:])

# betas_real = beta(epochs_np)

betas_real = xd_list[2:]

surf = ax_3d.plot_wireframe(B, 
                            E, 
                            plot_estimator_data_scaled, 
                            cmap = 'viridis', 
                            edgecolor = 'k', 
                            alpha = 0.8,
                            label = r"$|\mathcal{F} - \mathcal{M}|(\beta)$")

ax_3d.plot(betas_real, 
            epochs_np, 
            np.log10(np.array(memory_estimator_list[1:])), 
            'r-', 
            linewidth = 2, 
            label=r"$|\mathcal{F} - \mathcal{M}|(\beta(t))$")

ax_3d.set_xlabel(r"$\beta$", 
                 fontsize = 12)

ax_3d.set_ylabel("# Epochs", 
                 fontsize = 12)

ax_3d.set_zlabel("Estimator Value", 
                 fontsize = 12)

ax_3d.legend()

figure_total, axis_total = plt.subplots(dpi = 500)

axis_total.semilogy(total_error_list,
                    linestyle = ":",
                    alpha = 0.6,
                    label = r"Total error $(|\mathcal{F} - M|)$")

axis_total.semilogy(sum_error_list,
                    linestyle = "--",
                    label = r"$|\mathcal{F} - \mathcal{M}| + |\mathcal{M} - M|$")

axis_total.legend()

axis_total.set(title = "Error Comparation",
                xlabel = "# Epochs",
                ylabel = "Error value")

figure_xd, axis_xd = plt.subplots(dpi = 500)

axis_xd.plot(xd_list)

plt.show()
