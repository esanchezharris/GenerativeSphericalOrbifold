% dump_golden_solution.m -- reproduce the golden-dump run (my_man_maxp.mat,
% orbifold_type=4 => cones [4 2 2]) and save the CONVERGED embedding, so the
% Python port can be pinned against the reference's actual answer for the
% first time. Also re-saves A/b/wMat so cut determinism can be bit-checked
% against tests/golden/{A,b,wMat}.mat.
init;
load my_man_maxp.mat            % V 2502x3, T 5000x3, inds 3x1
cones = [4 2 2];
os = createOrbifoldStructure(V, T, cones, inds);
[boundary, cutter] = os.generate();
Vc = cutter.V; Tc = cutter.T;
fprintf('cut mesh: %d verts, %d faces\n', size(Vc,1), size(Tc,1));
L = cotmatrix(Vc, Tc);
s = Solver(L, boundary);
[x, xinit, XinitFlat, lg] = s.solve_bfgs_fast();
[A, b] = boundary.generateBoundaryEquations(length(s.adj));
[E_final, ~] = s.objective_and_grad(x);
fprintf('E_final (at normalized converged x): %.12f\n', E_final);
x_final = x; wMat = s.Wmat;
save('-v7', 'golden_solution_dump.mat', 'x_final', 'E_final', 'A', 'b', 'wMat');
fprintf('DUMP_OK golden_solution_dump.mat\n');
