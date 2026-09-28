
function [nNeumannBCs,nDirichletBCs,nNodes,nElements,elementDofs,x1,x2,globalStiffnessMatrix,globalForceVector,globalDisplacementVector]...
          = P4_SetUpFE(nDofsPerNode, nodalPositions, connectivities,NeumannBCs, DirichletBCs)

% we do not give the spatialDimensions anymore, we define the number of
% degrees of freedom per node in the input file

nNodes              = size(nodalPositions,1);               % number of global nodes
nDofs               = nNodes * nDofsPerNode;                % number of global dofs
nElements           = size(connectivities,1) ;              % number of elements
nNodesPerElement    = length(connectivities(1,1:3));        % number of nodes per element
nDofsPerElement     = nDofsPerNode*nNodesPerElement;        % number of dofs per element
nNeumannBCs         = size(NeumannBCs,1);                   % number of external loads applied
nDirichletBCs       = size(DirichletBCs,1);                 % number of Dirichlet boundary conditions

% elemental dofs
elementDofs = zeros(nElements,nDofsPerElement);

     for e = 1:nElements
         asmNode1 = connectivities(e,1);
         asmNode2 = connectivities(e,2);
         asmNode3 = connectivities(e,3);
         elementDofs(e,:) = [asmNode1*2-1, asmNode1*2,asmNode2*2-1, asmNode2*2,asmNode3*2-1, asmNode3*2];
     end

% extract x1 and x2 for global nodes
x1 = nodalPositions(:,1);
x2 = nodalPositions(:,2);

% ----------------------------------------------------------------------------------------------------------%

% initialise global vectors and matrices

globalStiffnessMatrix           = zeros(nDofs,nDofs);
globalForceVector               = zeros(nDofs,1);
globalDisplacementVector        = zeros(nDofs,1);

end