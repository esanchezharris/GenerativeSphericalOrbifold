classdef triangulation
    % Minimal Octave shim for MATLAB's triangulation object.
    % Only what this codebase uses on the sphere path: freeBoundary().
    properties
        T; V;
    end
    methods
        function obj=triangulation(T,V)
            obj.T=T; obj.V=V;
        end
        function fb=freeBoundary(obj)
            E=[obj.T(:,[1 2]);obj.T(:,[2 3]);obj.T(:,[3 1])];
            Es=sort(E,2);
            [ue,~,ic]=unique(Es,'rows');
            cnt=accumarray(ic,1);
            fb=ue(cnt==1,:);
        end
    end
end
