''' Action enums for ALE controller '''
import pygame

# ALE default enums
PLAYER_A_NOOP           = 0
PLAYER_A_FIRE           = 1
PLAYER_A_UP             = 2
PLAYER_A_RIGHT          = 3
PLAYER_A_LEFT           = 4
PLAYER_A_DOWN           = 5
PLAYER_A_UPRIGHT        = 6
PLAYER_A_UPLEFT         = 7
PLAYER_A_DOWNRIGHT      = 8
PLAYER_A_DOWNLEFT       = 9
PLAYER_A_UPFIRE         = 10
PLAYER_A_RIGHTFIRE      = 11
PLAYER_A_LEFTFIRE       = 12
PLAYER_A_DOWNFIRE       = 13
PLAYER_A_UPRIGHTFIRE    = 14
PLAYER_A_UPLEFTFIRE     = 15
PLAYER_A_DOWNRIGHTFIRE  = 16
PLAYER_A_DOWNLEFTFIRE   = 17


def action_map(k, game):
    ''' get a pygame_key and returns an ale action '''
    if game == "pong":
        if k[pygame.K_UP] or k[pygame.K_w]: return PLAYER_A_RIGHT
        elif k[pygame.K_DOWN] or k[pygame.K_s]: return PLAYER_A_LEFT
        else: return PLAYER_A_NOOP
    else:
        if k[pygame.K_SPACE]:  return PLAYER_A_FIRE
        elif k[pygame.K_UP] or k[pygame.K_w]: return PLAYER_A_UP
        elif k[pygame.K_DOWN] or k[pygame.K_s]: return PLAYER_A_DOWN
        elif k[pygame.K_LEFT] or k[pygame.K_a]: return PLAYER_A_LEFT
        elif k[pygame.K_RIGHT] or k[pygame.K_d]: return  PLAYER_A_RIGHT
        else: return PLAYER_A_NOOP
