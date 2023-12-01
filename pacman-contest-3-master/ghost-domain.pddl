
      (define (domain ghost)

          (:requirements
              :typing
              :negative-preconditions
          )

          (:types
              invaders cells
          )

          (:predicates
              (cell ?p)

              ;pacman's cell location
              (at-ghost ?loc - cells)

              ;food cell location
              (at-invader ?i - invaders ?loc - cells)

              ;Indicated if a cell location has a capsule
              (has-capsule ?loc - cells)

              ;connects cells
              (connected ?from ?to - cells)

          )

          ; move ghost towards the goal state of invader
          (:action move
              :parameters (?from ?to - cells)
              :precondition (and 
                  (at-ghost ?from)
                  (connected ?from ?to)
              )
              :effect (and
                          ;; add
                          (at-ghost ?to)
                          ;; del
                          (not (at-ghost ?from))       
                      )
          )

          ; kill invader
          (:action kill-invader
              :parameters (?loc - cells ?i - invaders)
              :precondition (and 
                              (at-ghost ?loc)
                              (at-invader ?i ?loc)
                            )
              :effect (and
                          ;; add

                          ;; del
                          (not (at-invader ?i ?loc))
                      )
          )
      )
      