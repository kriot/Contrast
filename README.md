#Алгоритм

Заданы цвета (тона) при помощи параметра H (HSV) или серый (определяется областью, независимой от H). 
Алгоритм: 
	1) Распределить каждый пиксель в одну из масок (или ни в какую из них)
	2) Раздуть и размыть маски
	3) Найти непокрытую область (outerMask)
	4) Повысить контрастность по каждой маске, оптимизируя: минимум в ноль, максимум в 255, средний цвет в тот, который мы хотим. OuterMask оставляет средний цвет на месте. Если контрастность увеличивается слишком сильно, уменьшить, сделав поворот функции обработки относительно средней по всем компанентам точке.
