using JLD
using LinearAlgebra
using PrettyTables
# ---
# Функция, которая вычисляет собственные числа и векторы.

function eigenstructure(A_::AbstractMatrix)#(A_::Array{Float64,2})
    A = copy(A_) # Это копия и плохо, потом нужно исправить

    eps_ = eps(1.0) # Пороговое значение пока равно машинному нулю, eps(1.0) = 2.220446049250313e-16
                    # Возможно, это нужно настроить --- в тестовом примере в evals, evecs
                    # пока остаются значения, например, в 3 раза больше машинного нуля.
                    # Может быть нужно делать, напрмер, eps_ = 10.0 * eps(1.0)    
    
    A[abs.(A) .<= eps_] .= 0.0 # зануляем все, что по модулю меньше машинного нуля

    evals, evecs = eigen(A)

    # Проверяем, что мнимые части собственных чисел и векторов --- малые
    # @assert проверяет, что условие в аргументе --- true, иначе обваливает программу.
    # Пока не понятно, что делать, если мнимые --- большие, додумаем, когда спокнемся тут в первый раз.
    @assert !any(x->abs(imag(x)) > eps_, evals) "There are evals with large imag component."
    @assert !any(x->abs(imag(x)) > eps_, evecs) "There are evecs with large imag component.", imag(evecs)
    
    # Если пришли сюда, то мнимые части собственных чисел и векторов --- малые,
    # просто берем действительные части.
    evals = real.(evals)
    evecs = real.(evecs)
    
    # Для конкретной матрицы в собственных числах остались значения меньше машинного нуля,
    # типа
    # ... 4.457265511633423e-17  -3.087546450961291e-17 -3.087546450961291e-17  0.0  0.0
    #     0.0 0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.7171161773626447e-18  9.560436158645797e-17
    #     9.002461448415954e-16
    # Зануляем их тут. И для собственных векторов --- также.
    
    #evals[abs.(evals) .< eps_] .= 0.0
    #evecs[abs.(evecs) .< eps_] .= 0.0
    
    # На всякий случай проверяем, что все собственные вектора на месте,
    # то есть ни один из них не занулился полностью.
    # Для этого считаем их норму и проверям.
    # Если все ок, то нормализуем каждый собственный вектор.
    
    col_norm = norm.(eachcol(evecs)) # Нормы слобцов.
    
    @assert all(col_norm .> eps_) "There are evecs with zero norm."
    
    evecs = l2_normalize_matrix_safe(evecs)
    #for ev in eachcol(evecs)
    #    ev = ev / norm(ev)
    #end
    evecs_left = Matrix{Float64}(I,13,13)
    try
    evecs_left = inv(evecs)
    catch e
    # Dump data
    save("./matrix.jld", "A", A_)
    # Handle the exception
    println("An error occurred: ", e) 
    exit(1)
    end
    # # Для каждого числа/вектора проверяем невязку, на тексте было 2e-15 -- 2e-14
    # for n in 1:28
    #    r = norm(evals[n] * evecs[:,n] - A * evecs[:,n])
    #    @printf("norm = %15.7e\n", r)
    #    end
    
    return evals, evecs, evecs_left
end # function eigenstructure

function eigenstructure_var2(A::AbstractMatrix)#(A_::Array{Float64,2})
    #A = copy(A_) # Это копия и плохо, потом нужно исправить

    eps_ = eps(1.0) # Пороговое значение пока равно машинному нулю, eps(1.0) = 2.220446049250313e-16
                    # Возможно, это нужно настроить --- в тестовом примере в evals, evecs
                    # пока остаются значения, например, в 3 раза больше машинного нуля.
                    # Может быть нужно делать, напрмер, eps_ = 10.0 * eps(1.0)    
    

    evals, evecs = eigen(A)
    

    # Проверяем, что мнимые части собственных чисел и векторов --- малые
    # @assert проверяет, что условие в аргументе --- true, иначе обваливает программу.
    # Пока не понятно, что делать, если мнимые --- большие, додумаем, когда спокнемся тут в первый раз.
    @assert !any(x->abs(imag(x)) > eps_, evals) "There are evals with large imag component."
    @assert !any(x->abs(imag(x)) > eps_, evecs) "There are evecs with large imag component.", imag(evecs)
    
    evals = real.(evals)
    evecs = real.(evecs)
    col_norm = norm.(eachcol(evecs)) # Нормы слобцов.
    
    @assert all(col_norm .> eps_) "There are evecs with zero norm."
    
    evecs = l2_normalize_matrix_safe(evecs)
    evecs_left = Matrix{Float64}(I,13,13)
    try
    evecs_left = inv(evecs)
    catch e
    # Dump data
    save("./matrix.jld", "A", A)
    # Handle the exception
    println("An error occurred: ", e) 
    exit(1)
    end
    
    return evals, evecs, evecs_left
end # function eigenstructure

function l2_normalize_matrix_safe(A::AbstractMatrix)
    """Нормирует сnjk,ws матрицы в L2, избегая деления на ноль."""
    normalized_A = similar(A)
    for i in axes(A, 2)
        col = A[:, i]
        col_norm = sqrt(dot(col,col))
        if col_norm ≈ 0
            normalized_A[:, i] .= 0  # Если норма нулевая, оставляем нули
        else
            normalized_A[:, i] = col ./ col_norm
        end
    end
    return normalized_A
end


function eigenstructure_cut(A::AbstractMatrix)#(A_::Array{Float64,2})
    #A = copy(A_) # Это копия и плохо, потом нужно исправить

    eps_ = eps(1.0) # Пороговое значение пока равно машинному нулю, eps(1.0) = 2.220446049250313e-16
                    # Возможно, это нужно настроить --- в тестовом примере в evals, evecs
                    # пока остаются значения, например, в 3 раза больше машинного нуля.
                    # Может быть нужно делать, напрмер, eps_ = 10.0 * eps(1.0)    
                    
  
    

    evals, evecs = eigen(A)
    print(evals)
    # Проверяем, что мнимые части собственных чисел и векторов --- малые
    # @assert проверяет, что условие в аргументе --- true, иначе обваливает программу.
    # Пока не понятно, что делать, если мнимые --- большие, додумаем, когда спокнемся тут в первый раз.
    @assert !any(x->abs(imag(x)) > eps_, evals) "There are evals with large imag component.", evals
    @assert !any(x->abs(imag(x)) > eps_, evecs) "There are evecs with large imag component.", evecs
    
    # Если пришли сюда, то мнимые части собственных чисел и векторов --- малые,
    # просто берем действительные части.
   
    evals = real.(evals)
    right = real.(evecs)
    sorted_indices = sortperm(evals)
    selected = vcat(sorted_indices[1:3], sorted_indices[11:13])
    right = right[:,selected]
    evals, evecs = eigen(A')

    
    @assert !any(x->abs(imag(x)) > eps_, evals) "There are evals with large imag component.", evals
    @assert !any(x->abs(imag(x)) > eps_, evecs) "There are evecs with large imag component.", evecs
        
    left = real.(evecs)
    sorted_indices = sortperm(evals)
    selected = vcat(sorted_indices[1:3], sorted_indices[11:13])
    left = left[:,selected]
    evals =  evals[selected]

    


    col_norm = norm.(eachcol(right)) # Нормы слобцов.
    
    @assert all(col_norm .> eps_) "There are right with zero norm."
    
    right = l2_normalize_matrix_safe(right)

    
    col_norm = norm.(eachcol(left)) # Нормы слобцов.
    coeff = zeros(6)    
    for i in 1:6
    	coeff[i] = dot(left[:,i],right[:,i])
    end
    @assert !any(x-> abs(x) < eps_, coeff) "There coeff = 0", coeff, evals
    
    for i in axes(left, 2)
        col = left[:, i]
        left[:, i] = col ./ coeff[i]
    end

#    @assert all(col_norm .> eps_) "There are left with zero norm."
    
#    c = zeros(6,6)   
#    for j in 1:6, i in 1:6
#    	c[i,j] = dot(left[:,i],right[:,j])
#    end
    
#    pretty_table(c)
    
    return evals, right, left'
end # function eigenstructure


# Тут конец кода для проверки собственных чисел
# ---
