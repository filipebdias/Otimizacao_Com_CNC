"""
Algoritmo Genético para Colocação 2D utilizando Retângulos Livres (Free Rectangles):
 - Mantém o tamanho do bounding box fixo (sheet_width x sheet_height).
 - Não permite redimensionamento das peças.
 - Suporta apenas uma única chapa (sem multi-chapas).
 - Peças que não couberem em nenhum retângulo livre, mesmo considerando rotações
   de 0° ou 90°, serão descartadas e penalizadas.
 - Rotação permitida: 0°/90° para peças retangulares e diamantes.
 - Penalização severa para peças descartadas.

O algoritmo é compatível com qualquer conjunto de peças (retangulares, diamantes, circulares etc.),
desde que não haja sobreposição ou cortes nas imagens.
"""

from common.layout_display import LayoutDisplayMixin
import random
import copy
import math
from typing import List, Dict, Any, Tuple

class GeneticAlgorithm(LayoutDisplayMixin):
    def __init__(
        self,
        TAM_POP: int,
        recortes_disponiveis: List[Dict[str, Any]],
        sheet_width: float,
        sheet_height: float,
        numero_geracoes: int = 100
    ):
        """
        Algoritmo Genético baseado em Retângulos Livres (Free Rectangles) para colocação 2D:
         - Não permite multi-chapas nem redimensionamento das peças.
         - Peças sem encaixe em retângulos livres serão descartadas e penalizadas.
         - Rotação disponível: 0° e 90° para peças retangulares e em formato de diamante.
        """
        print("Algoritmo Genético 2D Free-Rectangles - Sem redimensionar, sem multi-chapas, bounding box fixo.")
        self.TAM_POP = TAM_POP
        self.recortes_disponiveis = recortes_disponiveis
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.numero_geracoes = numero_geracoes

        # População: cada indivíduo é uma permutação de [0..n-1]
        self.POP: List[List[int]] = []
        self.best_individual: List[int] = []
        self.best_layout: List[Dict[str, Any]] = []
        self.best_fitness: float = float('inf')
        self.optimized_layout = None

        # Parâmetros do Algoritmo Genético
        self.mutation_rate = 0.1
        self.elitism = True

        self.initialize_population()

    # -------------------------------------------------------------------------
    # 1. Inicialização
    # -------------------------------------------------------------------------
    def initialize_population(self):
        n = len(self.recortes_disponiveis)
        base = list(range(n))
        for _ in range(self.TAM_POP):
            perm = base[:]
            random.shuffle(perm)
            self.POP.append(perm)

    # -------------------------------------------------------------------------
    # 2. Decodificação via Retângulos Livres
    # -------------------------------------------------------------------------
    def decode_layout(self, permutation: List[int]) -> Tuple[List[Dict[str,Any]], int]:
        """
        Transforma uma permutação de peças em um layout utilizando a técnica de Retângulos Livres (Free Rectangles).
         - A lista inicial de retângulos livres contém apenas a chapa completa (0,0, sheet_width, sheet_height).
         - Cada peça é testada em diferentes rotações (0°/90°) para encaixe em um retângulo livre disponível.
         - Caso a peça tenha um encaixe válido, ela é posicionada e os retângulos livres são atualizados.
         - Caso não haja encaixe possível, a peça é descartada e penalizada.
        
        Retorna: (layout gerado, quantidade de peças descartadas).
        """
        layout_result: List[Dict[str, Any]] = []
        free_rects: List[Tuple[float,float,float,float]] = []
        # Cada retângulo livre é (x, y, w, h)

        # Começamos com um retângulo livre do tamanho da chapa
        free_rects.append((0, 0, self.sheet_width, self.sheet_height))

        discarded = 0

        for idx in permutation:
            rec = self.recortes_disponiveis[idx]
            possible_configs = []
            if rec["tipo"] in ("retangular","diamante"):
                for rot in [0, 90]:
                    w,h = self.get_dims(rec, rot)
                    possible_configs.append((rot, w, h))
            else:
                w,h = self.get_dims(rec, 0)
                possible_configs.append((0, w, h))

            placed = False
            # Tenta cada configuração (rot)
            for (rot, w, h) in possible_configs:
                # Tenta colocar a peça em algum retângulo livre
                best_index = -1
                # Exemplo simples: escolhemos o primeiro retângulo onde caiba
                for i, (rx, ry, rw, rh) in enumerate(free_rects):
                    if w <= rw and h <= rh:
                        best_index = i
                        break
                if best_index != -1:
                    # Conseguiu colocar
                    placed = True
                    r_final = copy.deepcopy(rec)
                    r_final["rotacao"] = rot
                    # Posiciona a peça no canto superior-esquerdo do retângulo livre
                    (rx, ry, rw, rh) = free_rects[best_index]
                    r_final["x"] = rx
                    r_final["y"] = ry
                    layout_result.append(r_final)

                    # Atualiza retângulos livres (subtrai o espaço ocupado)
                    # Dividimos em até 2 novos retângulos livres:
                    #   1) Ao lado direito da peça
                    #   2) Abaixo da peça
                    # Remove o retângulo free_rects[best_index]
                    del free_rects[best_index]

                    # retângulo à direita
                    if w < rw:
                        newW = rw - w
                        if newW > 0:
                            free_rects.append((rx + w, ry, newW, rh))
                    # retângulo abaixo
                    if h < rh:
                        newH = rh - h
                        if newH > 0:
                            free_rects.append((rx, ry + h, w, newH))
                    break
            if not placed:
                # descartou a peça
                discarded += 1

        return (layout_result, discarded)

    def get_dims(self, rec: Dict[str,Any], rot: int) -> Tuple[float,float]:
        """Calcula e retorna as dimensões (largura, altura) da peça considerando seu tipo e a rotação aplicada (0° ou 90°)."""

        tipo = rec["tipo"]
        if tipo == "circular":
            d = 2*rec["r"]
            return (d, d)
        elif tipo in ("retangular","diamante"):
            if rot == 90:
                return (rec["altura"], rec["largura"])
            else:
                return (rec["largura"], rec["altura"])
        else:
            return (rec.get("largura",10), rec.get("altura",10))

    # -------------------------------------------------------------------------
    # 3. Avaliação (Fitness)
    # -------------------------------------------------------------------------
    def evaluate_individual(self, permutation: List[int]) -> float:
        """
        Avalia um indivíduo decodificando sua disposição no layout via Retângulos Livres (Free Rectangles).
        - O fitness é calculado com base na área final ocupada pelo bounding box resultante.
        - Peças descartadas recebem uma penalização significativa.
        """

        layout, discarded = self.decode_layout(permutation)

        # Calcula bounding box do layout
        if not layout:
            return self.sheet_width*self.sheet_height*2 + discarded*10000

        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')

        for rec in layout:
            angle = rec.get("rotacao", 0)
            w,h = self.get_dims(rec, angle)
            x0, y0 = rec["x"], rec["y"]
            x1, y1 = x0 + w, y0 + h
            x_min = min(x_min, x0)
            x_max = max(x_max, x1)
            y_min = min(y_min, y0)
            y_max = max(y_max, y1)

        area_layout = (x_max - x_min)*(y_max - y_min)
        penalty = discarded*10000
        return area_layout + penalty

    def evaluate_population(self):
        for perm in self.POP:
            fit = self.evaluate_individual(perm)
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_individual = perm[:]

    # -------------------------------------------------------------------------
    # 4. Operadores Genéticos (Permutações)
    # -------------------------------------------------------------------------
    def compute_fitness_scores(self) -> List[float]:
        fits = [self.evaluate_individual(perm) for perm in self.POP]
        return [1/(1+f) for f in fits]

    def roulette_selection(self) -> List[int]:
        scores = self.compute_fitness_scores()
        total = sum(scores)
        pick = random.random()*total
        current=0
        for perm, sc in zip(self.POP, scores):
            current+=sc
            if current>=pick:
                return perm
        return self.POP[-1]

    def crossover_two_point(self, p1: List[int], p2: List[int]) -> List[int]:
        size = len(p1)
        i1, i2 = sorted(random.sample(range(size),2))
        child = [None]*size
        child[i1:i2+1] = p1[i1:i2+1]
        p2_idx = 0
        for i in range(size):
            if child[i] is None:
                while p2[p2_idx] in child:
                    p2_idx+=1
                child[i] = p2[p2_idx]
                p2_idx+=1
        return child

    def mutate(self, perm: List[int]) -> List[int]:
        if random.random()<self.mutation_rate:
            i1,i2 = random.sample(range(len(perm)),2)
            perm[i1], perm[i2] = perm[i2], perm[i1]
        return perm

    def genetic_operators(self):
        new_pop = []
        if self.elitism and self.best_individual:
            new_pop.append(self.best_individual[:])
        while len(new_pop)<self.TAM_POP:
            p1 = self.roulette_selection()
            p2 = self.roulette_selection()
            child = self.crossover_two_point(p1,p2)
            child = self.mutate(child)
            new_pop.append(child)
        self.POP = new_pop[:self.TAM_POP]

    # -------------------------------------------------------------------------
    # 5. Loop Principal
    # -------------------------------------------------------------------------
    def run(self):
        for gen in range(self.numero_geracoes):
            self.evaluate_population()
            self.genetic_operators()
            if gen%10==0:
                print(f"Geração {gen} - Melhor Fitness: {self.best_fitness}")
        # Decodifica a melhor permutação
        layout, discarded = self.decode_layout(self.best_individual)
        self.optimized_layout = layout
        return self.optimized_layout

    def optimize_and_display(self):
        """
        Exibe o layout inicial (posições originais) e o layout final
        usando free rectangles (sem redimensionar ou multi-chapas).
        """
        self.display_layout(self.recortes_disponiveis, title="Initial Layout - GA")
        self.run()
        self.display_layout(self.optimized_layout, title="Optimized Layout - GA")
        return self.optimized_layout