function [globalDisplacementVector,globalForceVector] = P4_ApplyBCs_and_Solve(globalForceVector,globalDisplacementVector,globalStiffnessMatrix,...
                                                                              nNeumannBCs,NeumannBCs,nDirichletBCs,DirichletBCs,nDofsPerNode)

for loadIndex=1:nNeumannBCs
        dof = nDofsPerNode * NeumannBCs(loadIndex,1)-nDofsPerNode + NeumannBCs(loadIndex,2);
        globalForceVector(dof) = NeumannBCs(loadIndex,3);
end

% make a copy of K and zero out rows and add a 1 on the diagonal, where we have an essential bc 
K = globalStiffnessMatrix;

for boundIndex = 1:nDirichletBCs

        % Find essential boundary condition dof
        dof = nDofsPerNode * DirichletBCs(boundIndex,1)- nDofsPerNode + DirichletBCs(boundIndex,2);
        % Enforce essential boundary condition
        K(dof,:)   = 0;
        K(dof,dof) = 1.;
        
        globalForceVector(dof) =  DirichletBCs(boundIndex,3);
        globalDisplacementVector(dof) = DirichletBCs(boundIndex,3);
        
end

% solve for displacement:
globalDisplacementVector = K \ globalForceVector;

% solve for reaction forces:
globalForceVector = globalStiffnessMatrix*globalDisplacementVector;

end