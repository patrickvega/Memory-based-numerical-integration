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

deep_layers = 4
hidden_layers_dimension = 10

## ----------------------Training Parameters------------------##

batch_size = 1000
epochs = 5000
learning_rate = 10e-3
optimizer = "Adam"  # Adam or SGD

NN = Neural_Network(input_dimension = 1,
                    output_dimension = 1,
                    deep_layers = deep_layers,
                    hidden_layers_dimension = hidden_layers_dimension,
                    optimizer = optimizer,
                    learning_rate = learning_rate)

NN_memory = Neural_Network(input_dimension = 1,
                           output_dimension = 1,
                           deep_layers = deep_layers,
                           hidden_layers_dimension = hidden_layers_dimension,
                           optimizer = optimizer,
                           learning_rate = learning_rate)
 
params = NN.state_dict()

NN_memory.load_state_dict(params)

integration_points = torch.linspace(0, 1, 1000).unsqueeze(-1)

integration_nodes, integration_weights = leggauss(3)
integration_nodes = torch.tensor(integration_nodes, 
                                 requires_grad = False).unsqueeze(1)
integration_weights = torch.tensor(integration_weights,  
                                   requires_grad = False).unsqueeze(1)

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
 
    integral_value = torch.sum(mapped_weights.unsqueeze(-1) * nodes_value.unsqueeze(-1), dim = 1).squeeze()
    
    
    return integral_value

loss = lambda x: (NN.laplacian(x) + rhs(x))**2
loss_memory = lambda x: (NN_memory.laplacian(x) + rhs(x))**2
rhs = lambda x: pi**2 * torch.sin(pi * x)
exact = lambda x: torch.sin(pi * x)

beta = lambda t: exp(-0.001 * t) + 0.001

last_loss = 0.
last_loss_list = []
beta_list = []


def compute_loss(nb_iter, last_loss):
    
    x = torch.rand(batch_size)
    
    loss_function = torch.nn.L1Loss(reduction = 'mean')
        
    integrad_value = loss(x)
    integral_value_memory = loss_memory(x)
    
    exact_integral = integrate(loss, integration_points) # \mathcal{F}
    
        
    boundary_value = NN(torch.tensor([0., 1.]))**2
        
    res = torch.concat([integrad_value, boundary_value], axis = -1)     
    res_value = loss_function(res,torch.zeros_like(res,requires_grad = True))

    
    
    loss_value_memory = beta(nb_iter) * integral_value_memory + (beta(nb_iter) - 1) * last_loss
    exact_memory_term_integral = integrate(lambda x: beta(nb_iter) * loss_memory(x) + (beta(nb_iter) - 1) * last_loss, integration_points) # \mathcal{M}

    total_error = abs(torch.sum(exact_integral) - torch.mean(loss_value_memory))
    memory_error = abs(torch.sum(exact_integral) - torch.sum(exact_memory_term_integral)) 
    integration_error = abs(torch.sum(exact_memory_term_integral) - torch.mean(loss_value_memory))
        
    
    boundary_value_memory = NN_memory(torch.tensor([0., 1.]))**2 
    
    res_memory = torch.concat([loss_value_memory, boundary_value_memory], axis = -1)
    res_value_memory = loss_function(res_memory,torch.zeros_like(res_memory,requires_grad = True))

    return res_value, res_value_memory, total_error, memory_error, integration_error 

loss_list = []
loss_memory_list = []
total_error_list = []
memory_error_list = []
integration_error_list = []
sum_list = []

start_time = datetime.now()

print(f"{'='*30} Training {'='*30}")
for epoch in range(epochs):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{epochs} {'='*20}")
    
    loss_value, loss_value_memory, total_error, memory_error, integration_error =  compute_loss(epoch, last_loss)
    
    last_loss = loss_value_memory.item()
    
    loss_list.append(loss_value.item())
    loss_memory_list.append(last_loss)
    
    total_error_list.append(total_error.item())
    memory_error_list.append(memory_error.item())
    integration_error_list.append(integration_error.item())
    sum_list.append(memory_error.item() + integration_error.item())
    
    beta_list.append(beta(epoch))
    
    print(f"Loss: {loss_value.item():.8f} Loss with memory: {loss_value_memory.item():.8f}")
    
    NN.optimizer_step(loss_value)
    
    NN_memory.optimizer_step(loss_value_memory)
    
end_time = datetime.now()

execution_time = end_time - start_time

print(f"Training time: {execution_time}")

plot_points = torch.linspace(0, 1, 1000)

exact_evaluation = exact(plot_points)
exact_evaluation = exact_evaluation.cpu().detach().numpy()

NN_evaluation = NN(plot_points)
NN_evaluation = NN_evaluation.cpu().detach().numpy()

NN_memory_evaluation = NN_memory(plot_points)
NN_memory_evaluation = NN_memory_evaluation.cpu().detach().numpy()

plot_points = plot_points.cpu().detach().numpy()

figure_solution, axis_solution = plt.subplots(dpi=500)

axis_solution.plot(plot_points,
                   exact_evaluation,
                   alpha = 0.6,
                   label = "exact")

axis_solution.plot(plot_points,
                   NN_evaluation,
                   linestyle = "-.",
                   label = r"$u^{\theta}$")

axis_solution.plot(plot_points,
                   NN_memory_evaluation,
                   linestyle = ":",
                   label = r"$u_{M}^{\theta}$")

axis_solution.legend()

axis_solution.set(title = "PINNs final solution", 
                  xlabel = "x", 
                  ylabel = "u(x)")

figure_loss, axis_loss = plt.subplots(dpi=500)

axis_loss.semilogy(loss_list, 
                    linestyle = "-.", 
                    label = r"$\mathcal{L}^{\theta}$")

axis_loss.semilogy(loss_memory_list, 
                    linestyle = ":", 
                    label = r"$\mathcal{L}_{M}^{\theta}$")

axis_loss.legend()

axis_loss.set(title = "Loss Evolution", 
              xlabel = "# Epoch", 
              ylabel = "loss value")

# figure_beta, axis_beta = plt.subplots(dpi = 500)

# axis_beta.semilogy(beta_list)

# axis_beta.set(title = "Beta Evolution", 
#               xlabel = "# Epoch", 
#               ylabel = "Beta value")

figure_total, axis_total = plt.subplots(dpi = 500)

axis_total.semilogy(total_error_list,
                    linestyle = "-",
                    label = r"$|\mathcal{F}-M|$")
axis_total.semilogy(sum_list,
                    linestyle = ":",
                    label = r"$|\mathcal{F} - \mathcal{M}| + |\mathcal{M} - M|$")


# axis_total.semilogy(memory_error_list,
#                     linestyle = "--",
#                     label = r"$|\mathcal{F} - \mathcal{M}|$")

# axis_total.semilogy(integration_error_list,
#                     linestyle = "-.",
#                     label = r"$|\mathcal{M} - M|$")

axis_total.legend()

axis_total.set(title = "Total error")

# figure_loglog, axis_loglog = plt.subplots(dpi = 500)

# axis_loglog.loglog(sum_list, total_error_list)  

figure_error, axis_error = plt.subplots(dpi = 500)

axis_error.semilogy(memory_error_list,
                    linestyle = "-.",
                    label = r"$|\mathcal{F} - \mathcal{M}|$")

axis_error.semilogy(integration_error_list,
                    linestyle = ":",
                    label = r"$|\mathcal{M} - M|$")

axis_error.legend()

axis_error

plt.show()