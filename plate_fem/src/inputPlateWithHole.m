function [nodalPositions, connectivities, DirichletBCs, NeumannBCs, mprop] = inputPlateWithHole(h, sigma1, sigma2)
%INPUTPLATEWITHHOLEB
% Build a quarter plate-with-hole mesh, symmetry Dirichlet BCs, and
% mesh-consistent Neumann nodal loads obtained from boundary-edge traction
% integration.
%
% OUTPUTS
%   nodalPositions : [nNode x 2] node coordinates
%   connectivities : [nElem x 4] triangle connectivity + material id
%                    columns 1:3 are node ids, col 4 = material id
%   DirichletBCs   : [nDir x 3] rows [nodeID, dof, value]
%                    dof = 1 -> ux, dof = 2 -> uy
%   NeumannBCs     : [nNeu x 3] rows [nodeID, dof, value]
%                    equivalent nodal forces from edge traction integration
%   mprop          : [E, nu, t]
%
% INPUTS
%   h       : target mesh size near the hole
%   sigma1  : traction on right edge in x-direction
%   sigma2  : traction on top edge in y-direction

    % -----------------------------
    % Problem parameters
    % -----------------------------
    r = 1.0;          % hole radius
    L = 5.0;          % outer square size
    tol = 1e-6;       % geometric tolerance

    % Material properties [E, nu, thickness]
    mprop = [10.0, 0.3, 1.0];
    thickness = mprop(3);

    % -----------------------------
    % DistMesh path
    % -----------------------------
    if exist('distmesh2d', 'file') ~= 2
        thisDir = fileparts(mfilename('fullpath'));
        addpath(fullfile(thisDir, '..', 'distmesh'));
    end

    if exist('distmesh2d', 'file') ~= 2
        error('distmesh2d not found. Make sure DistMesh is on the MATLAB path.');
    end

    % -----------------------------
    % Geometry and mesh size
    % Domain = rectangle [0,L]x[0,L] minus circle centered at origin
    % -----------------------------
    fd = @(p) ddiff(drectangle(p, 0, L, 0, L), dcircle(p, 0, 0, r));

    % Smaller elements near the hole, larger away from the hole
    fh = @(p) h + ((0.5 - h) / 8.0) .* dcircle(p, 0, 0, r);

    % Fixed points to help preserve key boundary locations
    pfix = [ ...
        0.0, 5.0;
        5.0, 5.0;
        5.0, 0.0;
        r,   0.0;
        0.0, r;
        5.0, 0.5;
        5.0, 1.0;
        5.0, 1.5;
        5.0, 2.0;
        5.0, 2.5;
        5.0, 3.0;
        5.0, 3.5;
        5.0, 4.0;
        5.0, 4.5;
        5.0, 5.0;
        0.5, 5.0;
        1.0, 5.0;
        1.5, 5.0;
        2.0, 5.0;
        2.5, 5.0;
        3.0, 5.0;
        3.5, 5.0;
        4.0, 5.0;
        4.5, 5.0];
    pfix = unique(pfix, 'rows', 'stable');

    bbox = [0, 0; L, L];

    % DistMesh triangulation
    [nodalPositions, tri] = distmesh2d(fd, fh, h, bbox, pfix);

    % -----------------------------
    % Boundary node sets
    % -----------------------------
    left   = find(abs(nodalPositions(:,1) - 0.0) < tol);
    bottom = find(abs(nodalPositions(:,2) - 0.0) < tol);

    % -----------------------------
    % Dirichlet BCs
    % left edge   -> ux = 0
    % bottom edge -> uy = 0
    % Format: [nodeID, dof, value]
    % -----------------------------
    DirichletBCs = [ ...
        left,   ones(numel(left),1),   zeros(numel(left),1);
        bottom, 2*ones(numel(bottom),1), zeros(numel(bottom),1)];

    % Sort for cleanliness
    DirichletBCs = sortrows(DirichletBCs, [1 2]);

    % -----------------------------
    % Find all boundary edges
    % -----------------------------
    boundaryEdges = findBoundaryEdges(tri);

    % Select straight boundary edges
    rightEdges = selectBoundaryEdges(nodalPositions, boundaryEdges, 'right', L, tol);
    topEdges   = selectBoundaryEdges(nodalPositions, boundaryEdges, 'top',   L, tol);

    % -----------------------------
    % Assemble equivalent nodal force vector from tractions
    % -----------------------------
    nNode = size(nodalPositions, 1);
    F = zeros(2*nNode, 1);

    % Right boundary traction t = [sigma1; 0]
    if sigma1 ~= 0
        F = assembleEdgeTraction(F, nodalPositions, rightEdges, [sigma1; 0.0], thickness);
    end

    % Top boundary traction t = [0; sigma2]
    if sigma2 ~= 0
        F = assembleEdgeTraction(F, nodalPositions, topEdges, [0.0; sigma2], thickness);
    end

    % Convert global force vector into [nodeID, dof, value]
    NeumannBCs = globalForceToBCList(F);

    % -----------------------------
    % Append material id to connectivity
    % -----------------------------
    connectivities = [tri, ones(size(tri,1),1)];
end


% =========================================================================
% Helper functions
% =========================================================================

function boundaryEdges = findBoundaryEdges(tri)
%FINDBOUNDARYEDGES Return all edges belonging to only one triangle.
% Input:
%   tri : [nElem x 3]
% Output:
%   boundaryEdges : [nBndEdge x 2]

    allEdges = [ ...
        tri(:, [1 2]);
        tri(:, [2 3]);
        tri(:, [3 1])];

    allEdges = sort(allEdges, 2);

    [uniqueEdges, ~, ic] = unique(allEdges, 'rows');
    counts = accumarray(ic, 1);

    boundaryEdges = uniqueEdges(counts == 1, :);
end


function selectedEdges = selectBoundaryEdges(nodes, boundaryEdges, side, value, tol)
%SELECTBOUNDARYEDGES Select boundary edges on a straight side.
% side = 'right' or 'top'

    p1 = nodes(boundaryEdges(:,1), :);
    p2 = nodes(boundaryEdges(:,2), :);

    switch lower(side)
        case 'right'
            mask = abs(p1(:,1) - value) < tol & abs(p2(:,1) - value) < tol;

        case 'top'
            mask = abs(p1(:,2) - value) < tol & abs(p2(:,2) - value) < tol;

        otherwise
            error('Unknown side "%s". Use "right" or "top".', side);
    end

    selectedEdges = boundaryEdges(mask, :);
end


function F = assembleEdgeTraction(F, nodes, edges, traction, thickness)
%ASSEMBLEEDGETRACTION Assemble constant traction on 2-node linear edges.
%
% For one edge of length Le and constant traction t = [tx; ty],
% the consistent equivalent nodal force is:
%
%   fe = (Le * thickness / 2) * [tx; ty; tx; ty]

    if isempty(edges)
        return;
    end

    for e = 1:size(edges,1)
        n1 = edges(e,1);
        n2 = edges(e,2);

        x1 = nodes(n1, :);
        x2 = nodes(n2, :);
        Le = norm(x2 - x1);

        fe = (Le * thickness / 2.0) * [traction(1); traction(2); traction(1); traction(2)];

        dofs = [2*n1-1; 2*n1; 2*n2-1; 2*n2];
        F(dofs) = F(dofs) + fe;
    end
end


function NeumannBCs = globalForceToBCList(F)
%GLOBALFORCETOBCLIST Convert global force vector to [nodeID, dof, value].

    tolForce = 1e-14;
    idx = find(abs(F) > tolForce);

    if isempty(idx)
        NeumannBCs = zeros(0,3);
        return;
    end

    nodeID = ceil(idx / 2);
    dof = 2 - mod(idx, 2);   % odd -> 1, even -> 2
    dof(dof == 0) = 2;

    NeumannBCs = [nodeID, dof, F(idx)];
    NeumannBCs = sortrows(NeumannBCs, [1 2]);
end
