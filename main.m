clear; close all; clc;

%% network install

%layers sizes
input_size = 1;
hidden_size = 500;
output_size = 100;

% Activation function
func = @ActFuncs.Tanh;
%% network parmeters

gaama = 0.01;                   %W input-hidden scaling
delta = 0.98;                   %K scaling
connectivity = 0.35;            %connection ratio
alpha = 1;                      %leaking rate
beta = 1e-8;                    %regularization

%% input and Y0

time_steps = 10000;                     %number of time steps for input

input = randn(time_steps,1);            %creating white noise
Y0 = Y0_calc(input,output_size);        %calculating Y0

%% Weights init

%input to hidden
W_in = randn(input_size,hidden_size) .* gaama;

%hidden layer
K = k_init(hidden_size,connectivity,delta);             %creating K

%% feedforward

R = ff_ESN(input,hidden_size,output_size,time_steps,alpha,W_in,K,func);

%% Calculate optimal W
valid_steps = time_steps-output_size;

C = (R*R') * (1/valid_steps);
u = (R*Y0) * (1/valid_steps);

%with Ridge regulation
W_out = (C + beta.*eye(size(C))) \ u;

%% Network output
Y = func(R' * W_out);

R2_train = R2(Y,Y0,output_size);

%% test
%with a new set of white noise, we calculate new Y0 and using the old K and W-in
%and W-out
test_input = randn(time_steps,1);
Y0_test = Y0_calc(test_input,output_size);

R_test = ff_ESN(test_input,hidden_size,output_size,time_steps,alpha,W_in,K,func);
Y_test = func(R_test' * W_out);

R2_test = R2(Y_test,Y0_test,output_size);

%% plot and memory capcity

%plots
figure
plot(R2_train)
hold on
plot(R2_test)
legend('train','test')
title('R^2 as function of neuron index')
ylabel('R^2')
xlabel('neuron index')

%memory capcity
MC_train = sum(R2_train);
disp(['Memory capcity for train set: ',mat2str(MC_train,4),'%'])

MC_test = sum(R2_test);
disp(['Memory capcity for test set: ',mat2str(MC_test,4),'%'])
