# Gantt template for proposals (pgfgantt)

    make          # -> gantt.pdf
    make clean

Requires TeX Live with `pgfgantt` (in `texlive-pictures`).

## What to edit

| Cosa | Dove |
|---|---|
| Titolo e riferimento offerta | `\ProjTitle`, `\ProjRef` |
| Durata totale | `\Weeks` + il `{1}{32}` di `\begin{ganttchart}` + `\gantttitlelist{1,...,32}{1}` |
| Colori aziendali | blocco `\definecolor` |
| Larghezza colonne | `x unit` (0.62 cm = 32 settimane su A4 landscape) |
| Altezza righe | `y unit chart` |

## Comandi utili

    \ganttgroup{Fase}{inizio}{fine}          % barra di gruppo
    \ganttbar[name=id]{Attivita'}{da}{a}     % attivita'
    \ganttbar[progress=40]{...}{da}{a}       % con avanzamento
    \ganttmilestone{Nome}{settimana}         % milestone di progetto
    \custmilestone{Nome}{settimana}          % milestone a carico cliente
    \ganttlink{id1}{id2}                     % freccia di dipendenza
    \ganttvrule{Today}{18}                   % linea verticale (avanzamento)

Ogni riga va chiusa con `\\` tranne l'ultima prima dei `\ganttlink`.

## Per includerlo in un'offerta esistente

Togliere il preambolo e incollare il solo blocco `ganttchart` dentro un
`figure`, oppure compilare a parte e inserire il PDF:

    \includegraphics[width=\textwidth,angle=90]{gantt.pdf}

## Note

Le durate sono relative alla settimana 0 = ricezione ordine: cosi' l'allegato
non scade se il cliente firma mesi dopo. Le milestone a carico del cliente
vanno sempre esplicitate, altrimenti un loro ritardo diventa un ritardo
contrattuale del fornitore.
