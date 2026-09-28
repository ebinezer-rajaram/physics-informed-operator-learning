% compute element stiffness:
function [ElementStiffnessMatrix] = P4_ComputeStiffness(x1e,x2e, D, t)
   

me = [1,1,1;
      x1e(1),x1e(2),x1e(3);
      x2e(1),x2e(2),x2e(3)]';
      
    A = det(me)/2;  
  
    % construct Be
    Be_dx1(1,:) = [0 1 0]/me;
    Be_dx2(2,:) = [0 0 1]/me;
    
    Be = [ Be_dx1(1,1) 0 Be_dx1(1,2) 0 Be_dx1(1,3) 0;
          0 Be_dx2(2,1) 0 Be_dx2(2,2) 0  Be_dx2(2,3);  
          Be_dx2(2,1) Be_dx1(1,1)  Be_dx2(2,2) Be_dx1(1,2)  Be_dx2(2,3) Be_dx1(1,3)];
      
    ElementStiffnessMatrix = Be'*D*Be*A*t;
    
  
end