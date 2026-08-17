classdef graph
    % Minimal Octave shim for MATLAB's graph(): weighted undirected graph
    % from a sparse adjacency; shortestpath() via Dijkstra. Tie-breaking may
    % differ from MATLAB -- the regenerated A/b are bit-checked against the
    % golden dump to confirm the cut is identical.
    properties
        A;
    end
    methods
        function obj=graph(A)
            obj.A=A;
        end
        function [path,dist]=shortestpath(obj,s,t)
            n=size(obj.A,1);
            dist_v=inf(n,1); prev=zeros(n,1); visited=false(n,1);
            dist_v(s)=0;
            for it=1:n
                d=dist_v; d(visited)=inf;
                [du,u]=min(d);
                if isinf(du); break; end
                visited(u)=true;
                if u==t; break; end
                [nbr,~,w]=find(obj.A(:,u));
                for k=1:length(nbr)
                    v=nbr(k);
                    if ~visited(v) && du+w(k)<dist_v(v)
                        dist_v(v)=du+w(k); prev(v)=u;
                    end
                end
            end
            dist=dist_v(t);
            if isinf(dist); path=[]; return; end
            path=t;
            while path(1)~=s
                path=[prev(path(1)) path];
            end
        end
    end
end
