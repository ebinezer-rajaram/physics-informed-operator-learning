clc;clear;close all
% you do not need to change anything in this code
rng(1)
% ----------------------------------------------------------------------------------------------------------%
%                                       PRE-PROCESSING
% ----------------------------------------------------------------------------------------------------------%

scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, 'src'))
h =0.025;
sigma1 =0.1;
sigma2 =0;
[nodalPositions,connectivities,DirichletBCs,NeumannBCs,mprop] = inputPlateWithHole(h,sigma1,sigma2);

nDofsPerNode = 2;

Ey = mprop(1);
v  = mprop(2);
t  = mprop(3);

D = (Ey/(1-v^2))*[1 v 0
                  v 1 0
                  0 0 (1-v)/2];
% ----------------------------------------------------------------------------------------------------------%
%
% Use the variables (detailed below) for the computation in the remainder
% of script.
%
[nNeumannBCs,nDirichletBCs,nNodes,nElements,elementDofs,x1,x2,globalStiffnessMatrix,globalForceVector,globalDisplacementVector]  ...
    = P4_SetUpFE(nDofsPerNode, nodalPositions, connectivities,NeumannBCs, DirichletBCs);

% definition of the variables:
% nNeumannBCs         - number of external loads applied
% nDirichletBCs       - number of Dirichlet boundary conditions
% nNodes              - number of global nodes
% nElements           - number of elements
% elementDofs         - local-to-global map (connectivities) for the dofs
% x1, x2              - the x1 and x2 coordinates of the element nodes
% global -Force, -Stiffness, -Displacement matrices of the correct size,
% filled with zeros.

% ----------------------------------------------------------------------------------------------------------%
%                                                SOLVING
% ----------------------------------------------------------------------------------------------------------%

for e = 1:nElements
    
    asmNodes = connectivities(e,1:3);  % connectivity in terms of global nodes
    asmDofs  = elementDofs(e,:);       % connectivity in terms of global dofs
    mID      = connectivities(e,3);    % element material id
    

    x1e = x1(asmNodes);
    x2e = x2(asmNodes);

    ElementStiffnessMatrix = P4_ComputeStiffness(x1e,x2e, D, t);
  

    globalStiffnessMatrix(asmDofs, asmDofs) = globalStiffnessMatrix( asmDofs,asmDofs)...
                                                       + ElementStiffnessMatrix;
                                                                                                    
end
globalStiffnessMatrix;
[globalDisplacementVector,globalForceVector] = ...
  P4_ApplyBCs_and_Solve(globalForceVector,globalDisplacementVector,globalStiffnessMatrix,nNeumannBCs,NeumannBCs,nDirichletBCs,DirichletBCs,nDofsPerNode);

% ----------------------------------------------------------------------------------------------------------%
%                                        POST-PROCESSING
% ----------------------------------------------------------------------------------------------------------%


% Postprocessing and plotting  

strain = zeros( 3*nElements, 1);
stress = zeros( 3*nElements, 1);
stressvM = zeros(nElements,1);
stressvMB = zeros(nElements,1);
ICA_stress = zeros(nElements,3);
for i = 1:nElements
    ICA_stress(i,:) = [3*i-2 3*i-1 3*i];
end

for e = 1:nElements
    asm = connectivities(e,:);
    asm_K = elementDofs(e,:);
    asm_S = ICA_stress(e,:);
    x1e = x1(asm);
    x2e = x2(asm);
    
    me = [1,1,1;
          x1e(1),x1e(2),x1e(3);
          x2e(1),x2e(2),x2e(3)]';
    % construct Be
    Be_dx1(1,:) = [0 1 0]/me;
    Be_dx2(2,:) = [0 0 1]/me;
    
    Be = [ Be_dx1(1,1) 0 Be_dx1(1,2) 0 Be_dx1(1,3) 0;
          0 Be_dx2(2,1) 0 Be_dx2(2,2) 0  Be_dx2(2,3);  
          Be_dx2(2,1) Be_dx1(1,1)  Be_dx2(2,2) Be_dx1(1,2)  Be_dx2(2,3) Be_dx1(1,3)];
      
    strain_elm = Be*globalDisplacementVector(asm_K);
    stress_elm = D*strain_elm;
    stressvM_elm   = sqrt(stress_elm(1)^2 - stress_elm(1)*stress_elm(2) + stress_elm(2)^2 + 3*stress_elm(3)^2 );
    vonMisesStress = sqrt(stress_elm(1)^2  -stress_elm(1)*stress_elm(2) + stress_elm(2)^2 + 3*stress_elm(3)^2 );
    strain(asm_S,1)= strain_elm;
    stress(asm_S,1)= stress_elm;  
    stressvM(e) = stressvM_elm; 
    stressvMB(e) = vonMisesStress;
    stress_full(e,:) = stress_elm; 
    
     if x1e(1) == 0.3 && x2e(1) ==0
        fprintf('%s','magnitude von Mises stress (theta = 0) =')
        fprintf('\n')
        disp(stressvM_elm); 
    end
     if x1e(2) == 0.3 && x2e(2) == 0
        fprintf('%s','magnitude von Mises stress (theta = 0) = ')
        fprintf('\n')
        disp(stressvM_elm); 
     end
    if x1e(3) == 0.3 && x2e(3) == 0
        fprintf('%s','magnitude von Mises stress (theta = 0) = ')
        fprintf('\n')
        disp(stressvM_elm);
    end
end


u_ture = globalDisplacementVector(1:2:end); 
v_ture = globalDisplacementVector(2:2:end);

r= 1; 
error = 1e-5; 
p = nodalPositions;
p_full = p; 

L_boundary = p(abs(p(:,1)-0)<error,:);
T_boundary = p(abs(p(:,2)-5)<error,:); 
R_boundary = p(abs(p(:,1)-5)<error,:); 
B_boundary = p(abs(p(:,2)-0)<error,:);
C_boundary = p(abs(p(:,1).^2+p(:,2).^2-r.^2)<error,:);  
Boundary = [L_boundary; T_boundary; R_boundary; B_boundary;C_boundary]; 

C_index = find(abs(p(:,1).^2+p(:,2).^2-r.^2)<error); 
% =================================================
figure(1) 
hold on
% plot(L_boundary(:,1),L_boundary(:,2),'ro')
% plot(T_boundary(:,1),T_boundary(:,2),'ro')
% plot(R_boundary(:,1),R_boundary(:,2),'ro')
% plot(B_boundary(:,1),B_boundary(:,2),'ro')
% plot(C_boundary(:,1),C_boundary(:,2),'ro')
%plot(Boundary(:,1), Boundary(:,2), 'ro')
plot(p(:,1),p(:,2),'bo')

mask = any(ismember(p,Boundary),2); 
p(mask,:) = []; 
u_ture = globalDisplacementVector(1:2:end); 
v_ture = globalDisplacementVector(2:2:end);

u_data = u_ture(C_index,:);
v_data = v_ture(C_index,:); 

disp_data = [u_ture,v_ture];

figure(2);
hold on
axis([-1.0 6.0 -1.0 6.0]) %first two:x, second 2:y bounds
for i = 1:nElements
   asm = connectivities(i,1:3);
   x1_m = x1(asm);
   x2_m = x2(asm);
   t_m = stress_full(i,1);
   patch(x1_m,x2_m, t_m, 'EdgeColor', 'none')
end
title('Stress distribution \sigma_{11}')
grid off
colormap jet;
colorbar;

t = connectivities; 

t(:,4) = [];

save(fullfile(scriptDir, 'plate_data.mat'), 'L_boundary', 'T_boundary', 'R_boundary', 'B_boundary', 'C_boundary', 'Boundary', 'p', 't', 'p_full', 'C_index', 'disp_data')
% ============Explaination of the code ==============================% 
% 
% p_full: an [N_nodes, 2] matrix containing all the collocation points  

% t: an [N_element, 3] matrix containing the connectivity matrix of the
% FEM mesh

% p: a matrix containing the coordinates of collocation points that is not
% on the boundary. 

% L_boundary: collocation points at the left boundary
% R_boundary: collocation points at the right boundary
% T_boundary: collocation points at the top boundary
% B_boundary: collocation poitns at the bottom boundar
% C_boundary: collocation points near the hole of the plate 
