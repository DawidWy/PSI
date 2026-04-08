# Bartłomiej Waliłko, Dawid Wypych

**Prognoza parametrów fizycznych gwiazd z wykorzystaniem odwracalnych sieci neuronowych (cINN) na bazie danych SDSS**

# **1\. Wstęp i uzasadnienie projektu**

Współczesna astrofizyka obserwacyjna generuje ogromne ilości danych spektroskopowych, których ręczna interpretacja lub tradycyjna analiza numeryczna staje się niewydolna w obliczu milionów zarejestrowanych obiektów. Główną motywacją podjęcia niniejszego projektu jest rozwiązanie problemu wysokiego kosztu obliczeniowego, jaki niosą ze sobą klasyczne symulacje ewolucji i budowy gwiazd. Procesy te, oparte na modelowaniu transferu promieniowania w atmosferach gwiezdnych oraz dynamiki płynów w ich wnętrzach, wymagają użycia zaawansowanych klastrów obliczeniowych i trwają niejednokrotnie od kilku dni do wielu tygodni dla pojedynczego przypadku. Wprowadzenie modelu sztucznej inteligencji pełniącego rolę modelu zastępczego (surrogate model) pozwala na skrócenie tego czasu do ułamków sekund, co otwiera drogę do masowej klasyfikacji populacji gwiazdowych w naszej Galaktyce i pozwala na dynamiczne reagowanie na nowe odkrycia astronomiczne.

# **2\. Szczegółowe cele i parametry fizyczne**

Głównym celem badawczym jest stworzenie modelu, który na podstawie wejściowego wektora natężenia promieniowania będzie w stanie precyzyjnie wyznaczyć kluczowe parametry fizyczne gwiazdy. Projekt skupia się przede wszystkim na estymacji temperatury efektywnej (Teff​), grawitacji powierzchniowej (logg) oraz metaliczności (\[Fe/H\]), które determinują pozycję gwiazdy na diagramie Hertzsprunga-Russella. Dodatkowo, model ma dążyć do określenia parametrów trudniej uchwytnych, takich jak masa oraz wiek gwiazdy, co tradycyjnie wymaga skomplikowanego dopasowywania izochron ewolucyjnych. Celem projektu jest stworzenie narzędzia uniwersalnego, które po odpowiednim przeskalowaniu będzie mogło być stosowane do różnych typów widmowych, od chłodnych karłów typu M po gorące gwiazdy masywne.

# **3\. Charakterystyka bazy danych i preprocessing**

Fundamentem prac badawczych jest zbiór danych SDSS Science Archive Server w wersji DR19, dostarczający setki tysięcy wysokiej jakości widm optycznych. Praca z tak obszernym i surowym zbiorem danych wymaga zaawansowanego etapu przygotowawczego, który obejmuje przede wszystkim normalizację kontinuum widma w celu eliminacji wpływu poczerwienienia międzygwiazdowego oraz szumów instrumentalnych. Kluczowym wyzwaniem technicznym jest opracowanie algorytmów usuwających linie emisyjne pochodzące z ziemskiej atmosfery (tzw. sky lines) oraz przeprowadzenie resamplingu do wspólnej siatki długości fal, co jest niezbędne dla zachowania spójności wejścia sieci neuronowej. Prawidłowo przygotowany zbiór danych treningowych musi również uwzględniać augmentację w postaci sztucznego zaszumienia, aby model stał się odporny na zmienne warunki obserwacyjne panujące podczas rzeczywistych pomiarów.

# **4\. Architektura modelu: Odwracalne Sieci Neuronowe (cINN)**

Architektura Conditional Invertible Neural Networks adresuje tzw. problemy odwrotne, gdzie wiele stanów fizycznych może teoretycznie generować podobny sygnał obserwowalny. W klasycznej regresji sieci często zawodzą w obliczu degeneracji parametrów, podając uśrednione i fizycznie błędne wyniki, natomiast cINN dzięki swojej budowie opartej na odwracalnych blokach sprzęgających uczy się pełnej gęstości prawdopodobieństwa (posterior distribution). Pozwala to na mapowanie informacji w obie strony: od widma do parametrów oraz od parametrów do syntetycznego widma. Dzięki temu model nie tylko dostarcza konkretnych wartości fizycznych, ale również precyzyjnie szacuje niepewność predykcji, co w badaniach astrofizycznych jest równie istotne jak sam wynik liczbowy.

# **5\. Analiza wymagań, zasobów i ryzyk projektowych**

Skuteczna realizacja projektu może wymagać dostępu do jednostek GPU o pamięci VRAM nie przekraczającej 2GB, które umożliwią równoległe przetwarzanie wielowymiarowych wektorów danych spektralnych. Od strony programistycznej projekt opiera się na środowisku Python oraz frameworkach takich jak PyTorch, wspieranych przez specjalistyczne biblioteki astrofizyczne do manipulacji plikami w formacie FITS. Do głównych ryzyk projektowych należy obsługa obiektów typu "out-of-distribution", czyli gwiazd o ekstremalnie rzadkich cechach fizycznych, które mogą prowadzić do destabilizacji wyników modelu, co wymaga zaimplementowania mechanizmów wykrywania anomalii już na etapie inferencji.
