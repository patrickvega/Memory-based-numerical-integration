import matplotlib.pyplot as plt
import torch

from Neural_Network import Neural_Network
from datetime import datetime
from numpy import pi, exp 

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)

from numpy.polynomial.legendre import leggauss

## ----------------------Neural Network Parameters------------------##

deep_layers = 5
hidden_layers_dimension = 15

## ----------------------Training Parameters------------------##

batch_size = 1000
epochs = 5000
learning_rate = 10e-5
optimizer = "Adam"  # Adam or SGD

NN = Neural_Network(input_dimension = 1,
                    output_dimension = 1,
                    deep_layers = deep_layers,
                    hidden_layers_dimension = hidden_layers_dimension,
                    optimizer = optimizer,
                    learning_rate = learning_rate)

integration_points = torch.linspace(0, 1, 1000, requires_grad = False).unsqueeze(-1)

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

# rhs = lambda x: -20 * (torch.cos(20 * x) - x * torch.sin(20 * x))
rhs = lambda x: pi**2 * torch.sin(pi * x) 
f = lambda x: (NN.laplacian(x) + rhs(x))**2
# exact = lambda x: x * torch.sin(20 * x)
exact = lambda x: torch.sin(pi*x)

beta = lambda t: exp(-0.0005 * t)

M_previous = 0.

def compute_loss(t, M_previous):

    # Compute Values for Loss
    
    x = torch.rand(batch_size)
    
    loss_function = torch.nn.L1Loss(reduction = 'sum')
        
    F = torch.mean(f(x))
    
    bc = NN(torch.tensor([0., 1.]))**2
        
    loss = torch.concat([F.unsqueeze(-1), bc], axis = -1)     

    loss_value = loss_function(loss,torch.zeros_like(loss,requires_grad = True))

    # Compute Errors
    
    # with torch.no_grad():
            
    F_cal = integrate(f, integration_points)
    
    M = beta(t) * F + (1 - beta(t)) * M_previous

    M_cal = integrate(lambda x : beta(t) * f(x) + (beta(t) - 1) * M_previous, integration_points)

    total_error = abs(F_cal.item() - M.item())
    memory_error = abs(F_cal.item() - M_cal.item())
    integration_error = abs(M_cal.item() - M.item())

    return loss_value, M, F, F_cal, total_error, memory_error, integration_error
    
    return loss_value

loss_list = []
M_list = []
F_list = []
F_cal_list = []
total_error_list = []
memory_error_list = []
integration_error_list = []
sum_error_list = []


start_time = datetime.now()

print(f"{'='*30} Training {'='*30}")
for epoch in range(epochs):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{epochs} {'='*20}")
        
    loss_value, M, F, F_cal, total_error, memory_error, integration_error,  = compute_loss(epoch, M_previous)
    
    M_previous = M.item()
    
    loss_list.append(loss_value.item())
    M_list.append(M.item())
    F_list.append(F.item())
    F_cal_list.append(F_cal.item())
    total_error_list.append(total_error)
    memory_error_list.append(memory_error)
    integration_error_list.append(integration_error)
    sum_error_list.append(memory_error + integration_error)
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

figure_solution, axis_solution = plt.subplots()

axis_solution.plot(plot_points,
                   exact_evaluation,
                   alpha = 0.6,
                   label = "exact")

axis_solution.plot(plot_points,
                   NN_evaluation,
                   linestyle = "-.",
                   label = r"$u^{\theta}$")

axis_solution.legend()

axis_solution.set(title = "PINNs final solution", 
                  xlabel = "x", 
                  ylabel = "u(x)")

figure_loss, axis_loss = plt.subplots()

axis_loss.semilogy(loss_list)

axis_loss.set(title = "Loss Evolution", 
              xlabel = "# Epoch", 
              ylabel = "loss value")

figure_memory, axis_memory = plt.subplots()


axis_memory.semilogy(F_list,
                     linestyle = "-.",
                     label = r"$F$")

axis_memory.semilogy(M_list,
                     linestyle = ":",
                     label = r"$M$")

axis_memory.semilogy(F_cal_list,
                      linestyle = "-",
                      alpha = 0.6,
                      color = "black",
                      label = r"$\mathcal{F}$")

axis_memory.legend()

figure_total, axis_total = plt.subplots()

axis_total.semilogy(total_error_list,
                    linestyle = ":",
                    label = r"$|\mathcal{F} - M|$")

axis_total.semilogy(sum_error_list,
                    linestyle = ":",
                    label = r"$|\mathcal{F} - M| + |\mathcal{M} - M|$")

axis_total.legend()

figure_error, axis_error = plt.subplots()

axis_error.semilogy(memory_error_list,
                    linestyle = ":",
                    label = r"$|\mathcal{F} - M|$")

axis_error.semilogy(integration_error_list,
                    linestyle = "-.",
                    label = r"$|\mathcal{M} - M|$")

axis_error.legend()

# figure_loglog, axis_loglog = plt.subplots()

# axis_loglog.loglog(F_list,M_list)

plt.show()

  