# Structure de l'Équipe Scrum

## Diagramme de l'Équipe

```mermaid
graph TB
    subgraph "Product Owners"
        PO[Youssef<br/>Product Owner]
    end
    
    subgraph "Stakeholders"
        S1[Wassim<br/>Stakeholder]
        S2[Aziza<br/>Stakeholder]
    end
    
    subgraph "Team 1 - Équipe de Développement"
        T1[amel<br/>Développeur]
        T2[chedia<br/>Développeur]
        T3[rayen<br/>Développeur]
    end
    
    PO -->|Priorise les besoins| T1
    PO -->|Priorise les besoins| T2
    PO -->|Priorise les besoins| T3
    
    S1 -->|Fournit les exigences| PO
    S2 -->|Fournit les exigences| PO
    
    S1 -.->|Feedback| T1
    S1 -.->|Feedback| T2
    S1 -.->|Feedback| T3
    S2 -.->|Feedback| T1
    S2 -.->|Feedback| T2
    S2 -.->|Feedback| T3
    
    T1 <-->|Collaboration| T2
    T2 <-->|Collaboration| T3
    T1 <-->|Collaboration| T3
    
    style PO fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    style S1 fill:#9E9E9E,stroke:#616161,stroke-width:2px,color:#fff
    style S2 fill:#9E9E9E,stroke:#616161,stroke-width:2px,color:#fff
    style T1 fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
    style T2 fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
    style T3 fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
```

## Structure Hiérarchique

```mermaid
graph TD
    PO[Youssef<br/>Product Owner]
    
    subgraph "Stakeholders"
        S1[Wassim]
        S2[Aziza]
    end
    
    subgraph "Équipe de Développement"
        T1[amel]
        T2[chedia]
        T3[rayen]
    end
    
    S1 --> PO
    S2 --> PO
    PO --> T1
    PO --> T2
    PO --> T3
    
    style PO fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    style S1 fill:#9E9E9E,stroke:#616161,stroke-width:2px,color:#fff
    style S2 fill:#9E9E9E,stroke:#616161,stroke-width:2px,color:#fff
    style T1 fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
    style T2 fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
    style T3 fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
```

## Rôles et Responsabilités

### 👤 Product Owner
**Youssef**
- Définit et priorise le Product Backlog
- Communique la vision du produit
- Valide les fonctionnalités développées
- Représente les besoins des stakeholders

### 👥 Stakeholders
**Wassim & Aziza**
- Fournissent les exigences métier
- Donnent du feedback sur les livrables
- Valident les fonctionnalités
- Participent aux démonstrations (Sprint Review)

### 👨‍💻 Équipe de Développement
**amel, chedia, rayen**
- Développent les fonctionnalités
- Estiment les tâches
- Participent aux cérémonies Scrum
- S'auto-organisent pour atteindre les objectifs du Sprint

## Cérémonies Scrum

1. **Sprint Planning** : PO + Équipe
2. **Daily Scrum** : Équipe uniquement
3. **Sprint Review** : PO + Équipe + Stakeholders
4. **Sprint Retrospective** : Équipe uniquement

## Flux de Communication

```
Stakeholders (Wassim, Aziza)
    ↓ [Exigences & Feedback]
Product Owner (Youssef)
    ↓ [User Stories & Priorités]
Équipe de Développement (amel, chedia, rayen)
    ↓ [Livrables]
Product Owner (Youssef)
    ↓ [Validation]
Stakeholders (Wassim, Aziza)
```
