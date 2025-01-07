#IMPORTACIÓN DE LIBRERÍAS
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

class infiniteBox:
    """
    Clase que simula un sistema de dos partículas bosónicas o fermiónicas (o una sola partícula) y calcula probabilidades, funciones de onda,
    factores de Boltzmann y distribuciones térmicas, en función de la temperatura y otros parámetros físicos.
    
    Atributos:
    ----------
    temperatura : float
        Temperatura adimensional del sistema.
    """
    
    def __init__(self, t):
        self.temperatura = t

    def phi(self, n, x):
        """
        Calcula los autoestados de la base (phi) asociadas a la función de onda total de la(s) partícula(s).

        Parámetros:
        -----------
        n : int
            Índice del nivel energético.
        x : float
            Posición adimensional de la partícula.

        Retorna:
        --------
        float
            Valor de la n-ésima función propia de la base, en la posición x.
        """
        c = np.sqrt(2 / np.pi) #Factor de normalización
        return c * np.sin(n * x) #Autofunción de acuerdo a la ecuación (I)
    
    def psi(self, n, x, mode):
        """
        Calcula la función de onda de dos partículas dependiendo de si es fermión o bosón

        Parámetros:
        -----------
        n : list
            Lista de índices de los niveles energéticos para cada partícula [n1, n2].
        x : list
            Lista de posiciones espaciales de las partículas [x1, x2].
        mode : str
            Tipo de sistema: 'fermion' para partículas fermiónicas, 'boson' para partículas bosónicas.

        Retorna:
        --------
        float
            Valor de la función de onda para las partículas en las posiciones x.
        """
        c = np.sqrt(1/2) #Factor de normalización
        term1 = self.phi(n[0], x[0]) * self.phi(n[1], x[1]) #Primer término de autofunciones
        term2 = self.phi(n[0], x[1]) * self.phi(n[1], x[0]) #Segundo término de autofunciones
        
        if mode == 'fermion':
            Total = c * (term1 - term2) #Si la partícula es un bosón, la función de onda es antisimétrica
        elif mode == 'boson':
            Total = c * (term1 + term2) #Si la partícula es un fermión, la función de onda es simétrica
        
        return Total #Se retorna la función de onda correspondiente, dependiendo del caso
    
    def P(self, n, x, mode, spin=False):
        """
        Calcula la probabilidad de encontrar dos partículas en ciertas posiciones, con o sin spin.

        Parámetros:
        -----------
        n : list
            Lista de índices de los niveles energéticos para cada partícula [n1, n2].
        x : list
            Lista de posiciones de las partículas [x1, x2].
        mode : str
            Tipo de sistema: 'fermion' para partículas fermiónicas, 'boson' para partículas bosónicas.
        spin : bool, opcional
            Si se considera el efecto del spin (default: False).

        Retorna:
        --------
        float
            Valor de la probabilidad calculada.
        """
        if spin == False: #Si NO se considera el spin...
            p = self.psi(n, x, mode) * self.psi(n, x, mode) #...La probabilidad será el modulo al cuadrado de la función de onda
        else: #SI se considera el spin...
            term1 = self.psi(n, x, 'boson') * self.psi(n, x, 'boson') / 2 #...Se debe tener en cuenta la parte simétrica
            term2 = self.psi(n, x, 'fermion') * self.psi(n, x, 'fermion') / 2 #La parte antisimétrica
            p = (term1 / 4) + (3 * term2 / 4) #Y sus contribuciones a la probabilidad
        
        return p
    
    def BoltzmannFactor(self, n):
        """
        Calcula el factor de Boltzmann para un nivel energético dado.

        Parámetros:
        -----------
        n : int o list
            Índice (o lista de índices) de los niveles energéticos.

        Retorna:
        --------
        float
            Factor de Boltzmann para el nivel energético especificado.
        """
        term = (np.linalg.norm(n))**2 / self.temperatura #Término de energía intrínseca vs térmica
        return np.exp(-term) #Se calcula el factor de Boltzmann de acuerdo a la teoría
    
    def P_th(self, x, nmax, mode='default', spin=False):
        """
        Calcula la probabilidad térmica normalizada considerando las distribuciones de Boltzmann.

        Parámetros:
        -----------
        x : list o float
            Posiciones espaciales de las partículas. Puede ser un único valor o una lista [x1, x2].
        nmax : int
            Máximo índice del nivel energético a considerar en los cálculos.
        mode : str, opcional
            'default' para sistemas clásicos o modos específicos ('boson', 'fermion') (default: 'default').

        Retorna:
        --------
        float
            Probabilidad térmica normalizada para el sistema en las posiciones especificadas.
        """
        #Se inicializan las sumatorias
        suma1 = 0
        suma2 = 0
        
        #La probabilidad termalizada también debe ser separada en el caso de "con spin" o "sin spin" 
        if spin == False: #Acá están los tres casos en los que no se tiene cuenta el spin

            if mode == 'default': #Si estamos hablando de una sola partícula...
                for i in range(1, nmax+1):
                    p = self.phi(i, x) * self.phi(i, x) #...Usamos la probabilidad unidimensional
                    term1 = self.BoltzmannFactor(i) * p #Este término representa el numerador de la sumatoria
                    suma1 += term1
                    suma2 += self.BoltzmannFactor(i) #Este término representa el denominador de la sumatoria
                    
                Total = suma1 / suma2 #Se llega a la probabilidad termalizada unidimensional

            elif mode == 'fermion': #Si estamos hablando de las dos partículas fermiónicas...
                for i in range(1, nmax):
                    for j in range(i+1, nmax+1): #NO se cuentan estados repetidos
                        term1 = self.BoltzmannFactor([i, j]) * self.P([i, j], [x[0], x[1]], mode) #...Se usa la probabilidad 2-dimensional
                        suma1 += term1 #Se añade el término del numerador a la sumatoria
                        suma2 += self.BoltzmannFactor([i, j]) #Este término representa el denominador de la sumatoria
                    
                Total = suma1 / suma2 #Se llega a la probabilidad termalizada 2-dimensional

            elif mode == 'boson': #Si estamos hablando de las dos partículas bosónicas...
                for i in range(1, nmax):
                    for j in range(i, nmax+1): #SI se cuentan estados repetidos
                        term1 = self.BoltzmannFactor([i, j]) * self.P([i, j], [x[0], x[1]], mode) #...Se usa la probabilidad 2-dimensional
                        suma1 += term1 #Se añade el término del numerador a la sumatoria
                        suma2 += self.BoltzmannFactor([i, j]) #Este término representa el denominador de la sumatoria
                    
                Total = suma1 / suma2 #Se llega a la probabilidad termalizada 2-dimensional
        
        if spin == True: #Acá está el caso en el que se tiene en cuenta el spin

            for i in range(1, nmax):
                    for j in range(i+1, nmax+1):
                        term1 = self.BoltzmannFactor([i, j]) * self.P([i, j], [x[0], x[1]], 'fermion', spin) #...Se usa la probabilidad 2-dimensional
                        suma1 += term1 #Se añade el término del numerador a la sumatoria
                        suma2 += self.BoltzmannFactor([i, j]) #Este término representa el denominador de la sumatoria
                    
            Total = suma1 / suma2 #Se llega a la probabilidad termalizada 2-dimensional


        return Total
    
class finiteBox:#(infiniteBox):
    def __init__(self, t):
        # super().__init__(t)
        t=0
        self.temperatura = t
    
    def find_roots(self, f, R_a, symmetry):
        # Detectar si un valor está cerca de una asíntota vertical de tan(alpha)
        def is_near_asymptote(alpha, threshold=0.1):
            if symmetry:
                return np.any(np.isclose(alpha, (np.pi / 2) + np.pi * np.arange(0, 10), atol=threshold))
            else:
                return np.any(np.isclose(alpha, np.pi + np.pi * np.arange(0, 10), atol=threshold))
            
        # Parámetros de búsqueda
        alpha_start = 0.1  # Comienzo del rango
        alpha_end = R_a - 0.1  # Fin del rango (R ligeramente menos para evitar raíz inválida)
        num_intervals = 1000  # Número de subintervalos
        tolerance = 1e-1  # Tolerancia para encontrar raíces

        # Dividimos el rango en subintervalos
        alpha_vals = np.linspace(alpha_start, alpha_end, num_intervals)
        f_vals = []

        # Calculamos los valores de f(alpha), excluyendo las asíntotas
        for alpha in alpha_vals:
            if is_near_asymptote(alpha):
                f_vals.append(np.nan)  # Marcar como indefinido
            else:
                f_vals.append(f(alpha, R_a))

        # Visualizamos la función
        # plt.plot(alpha_vals, f_vals, 'k-', label=f'f(α), R={R}')
        # plt.axhline(0, color='red', linestyle='--', label='y = 0')
        # plt.axvline(np.pi/2, color='blue', linestyle='--', label='$x = \pi/2$')
        # plt.axvline(3*np.pi/2, color='blue', linestyle='--', label='$x = 3\pi/2$')
        # plt.xlabel('α')
        # plt.ylabel('f(α)')
        # plt.ylim(-20,20)
        # plt.title('Búsqueda de todas las raíces de f(α)')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # Buscamos raíces en los subintervalos
        roots = []
        for i in range(len(alpha_vals) - 1):
            a, b = alpha_vals[i], alpha_vals[i + 1]
            if np.isnan(f_vals[i]) or np.isnan(f_vals[i + 1]):  # Saltar intervalos no definidos
                continue
            if f_vals[i] * f_vals[i + 1] < 0:  # Cambio de signo
                # Verificamos si el intervalo está cerca de una asíntota
                if is_near_asymptote(a) or is_near_asymptote(b):
                    continue
                root_result = root_scalar(f, args=(R_a,), bracket=[a, b], method='bisect', xtol=tolerance)
                if root_result.converged:
                    root = root_result.root
                    roots.append(root)
        
        return roots
        
        # Mostramos las raíces encontradas
        # print(f"Se encontraron {len(roots)} raíces:")
        # for i, root in enumerate(roots, start=1):
        #     print(f"Raíz {i}: α_n = {root:.6f}")
    
    def phi(self, A, a, u, x, symmetry):
        
        def outside(x, side="l"):
            if side == "l":
                return A*np.exp(u*x)
            elif side == "r":
                return A*np.exp(-u*x)

        def inside(x):
            if symmetry:
                return A*np.cos(a*x)
            else:
                return A*np.sin(a*x)



        if symmetry:
            func = np.piecewise( x, #Variable dependiente
                                [x < -1, (x >= -1) & (x <= 1), x > 1], #Tramos de la función
                                [lambda x: outside(x), lambda x: inside(x), lambda x: outside(x, "r")] ) #Funciones correspondientes

        else:
            func = np.piecewise( x, #Variable dependiente
                                [x < -1, (x >= -1) & (x <= 1), x > 1], #Tramos de la función
                                [lambda x: outside(x), lambda x: inside(x), lambda x: -outside(x, "r")] ) #Funciones correspondientes

        return func

    def psi(self, A, a, u, x, mode, symmetry):
        #c = np.sqrt(1/2) #Factor de normalización
        #Para el caso simétrico
        term1 = self.phi(A, a[0], u, x[0], symmetry) * self.phi(A, a[1], u, x[1], symmetry) #Primer término de autofunciones
        term2 = self.phi(A, a[0], u, x[1], symmetry) * self.phi(A, a[1], u, x[0], symmetry) #Segundo término de autofunciones
        
        # #Para el caso asimétrico
        # term1 = self.phi(A, a[0], u, x[0], symmetry=False) * self.phi(A, a[1], u, x[1], symmetry=False) #Primer término de autofunciones
        # term2 = self.phi(A, a[0], u, x[1], symmetry=False) * self.phi(A, a[1], u, x[0], symmetry=False) #Segundo término de autofunciones


        if mode == 'fermion':
            Total = A * (term1 - term2) #Si la partícula es un bosón, la función de onda es antisimétrica
        elif mode == 'boson':
            Total = A * (term1 + term2) #Si la partícula es un fermión, la función de onda es simétrica
        
        return Total #Se retorna la función de onda correspondiente, dependiendo del caso
    
    def P(self, A, a, u, x, mode, symmetry, spin=False):
        
        #Caso simétrico
        # if symmetry:
        if spin == False: #Si NO se considera el spin...
            p = self.psi(A, a, u, x, mode, symmetry) * self.psi(A, a, u, x, mode, symmetry) #...La probabilidad será el modulo al cuadrado de la función de onda
        else: #SI se considera el spin...
            term1 = self.psi(A, a, u, x, 'boson', symmetry) * self.psi(A, a, u, x, 'boson', symmetry) / 2 #...Se debe tener en cuenta la parte simétrica
            term2 = self.psi(A, a, u, x, 'fermion', symmetry) * self.psi(A, a, u, x, 'fermion', symmetry) / 2 #La parte antisimétrica
            p = (term1 / 4) + (3 * term2 / 4) #Y sus contribuciones a la probabilidad

        # #Caso asimétrico
        # else:
        #     if spin == False: #Si NO se considera el spin...
        #         p = self.psi(A, a, u, x, mode, spin, ) * self.psi(A, a, u, x, mode) #...La probabilidad será el modulo al cuadrado de la función de onda
        #     else: #SI se considera el spin...
        #         term1 = self.psi(A, a, u, x, 'boson') * self.psi(A, a, u, x, 'boson') / 2 #...Se debe tener en cuenta la parte simétrica
        #         term2 = self.psi(A, a, u, x, 'fermion') * self.psi(A, a, u, x, 'fermion') / 2 #La parte antisimétrica
        #         p = (term1 / 4) + (3 * term2 / 4) #Y sus contribuciones a la probabilidad

        
        return p

    def BoltzmannFactor(self, n):
        pass

    def P_th(self, x, nmax, mode='default', spin=False):
        pass



class Plotter:
    def __init__(self, dim, title):
        self.dimension = dim
        self.titulo = title

    def plot(self, x, f, x_size, y_size, xlabel, ylabel, label ='', elev=0, azim=0, z_lim=0):
        
        if self.dimension == 1:

            plt.figure(figsize=(x_size, y_size))
            for func, lbl in zip(f, label):
                plt.plot(x, func, label = lbl)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(self.titulo)
            plt.legend()
            plt.tight_layout()
            plt.grid()
            plt.show()
        
        elif self.dimension == 2:

            fig = plt.figure(figsize=(x_size, y_size))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot_surface(x[0], x[1], f, cmap='viridis', edgecolor='k', linewidth=0.5, alpha=0.9)
            ax.set_title(self.titulo, fontsize=16, pad=20)
            ax.set_xlabel(xlabel[0], fontsize=12, labelpad=10)
            ax.set_ylabel(xlabel[1], fontsize=12, labelpad=10)
            ax.set_zlabel(ylabel, fontsize=12, labelpad=10)
            ax.set_zlim(z_lim)

            # Cambiar la orientación de la vista
            ax.view_init(elev=elev, azim=azim)  # Puedes ajustar los valores de elev y azim como desees

            ax.grid()
            ax.set_box_aspect([1, 1, 0.6])

            plt.tight_layout()
            plt.show()