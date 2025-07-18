using LinearAlgebra
using Printf
using JLD

# На тесте получалось, что в собственных числах есть малые мнимые части,
# но матрицы t и z --- действительные.


# Вариант 1:
#     1. Занулить малые элементы в матрице t
#     2. Вычислить A2 = z * t * z'
#     3. Применить eigen к A2
#
# В результате, на тесте: собственные числа имеют мнимую часть 1e-17, собственные вектора имеют мнимую часть 0.0 точно.


# Вариант 2:
# 1. В матрице t зануляем элементы меньшие машинного нуля
# 2. Вычисляем собственные числа l и векторы [y] матрицы t -> на тесте получились только действительные значения
# 3. Вычисляем собственные векторы матрицы A как evecs = z * y

# Результат очевидно чисто действительне собственные числа и векторы.    

function eigenstructure_schur(A::AbstractMatrix)

    eps_ = eps(1.0) # Пороговое значение пока равно машинному нулю, eps(1.0) = 2.220446049250313e-16
                    # Возможно, это нужно настроить --- в тестовом примере в evals, evecs
                    # пока остаются значения, например, в 3 раза больше машинного нуля.
                    # Может быть нужно делать, напрмер, eps_ = 10.0 * eps(1.0)    
    
    eps_ = 10.0 * eps_

    
    D = schur(A)
    t, z, _ = D # D.values не используем, тут они пока комплексные с малыми мнимыми частями,
                # t и z --- гарантированно действительные (Float64)

    t[abs.(t) .< eps_] .= 0.0
    @assert istriu(t) "Matrix t is not upper triangular"
    
    evals_t, evecs_t = eigen(t)

    # eigen() отдает всегда комплексные числа, поэтому контролируем и убираем мнимую часть
    @assert any(imag(evals_t) .< eps_) "There are evals_t with large imag component."
    @assert any(imag(evecs_t) .< eps_) "There are evecs_t with large imag component."

    # Теперь можно занулять брать действительную часть
    evals_t = real(evals_t)
    evecs_t = real(evecs_t)
    
    evals = evals_t
    evecs = z * evecs_t

    # # Для каждого числа/вектора проверяем невязку, на тексте было 2e-16 -- 2e-15
    # @printf("\n")
    # @printf("eigenstructure_schur:\n")
    # for n in 1:28
    #    r = norm(evals[n] * evecs[:,n] - A * evecs[:,n])
    #    @printf("n = %3i, norm = %15.7e\n", n, r)
    # end

    # TODO:
    #     Далье можно снова проверить невырожденность и нормализовать. 
    
    return evals, evecs
        
        
end # function eigenstructure_schur(A::Array{Float64,2})


#evals, evecs = eigenstructure_schur(A)
