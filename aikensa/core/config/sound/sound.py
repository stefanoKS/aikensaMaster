import pygame

pygame.mixer.init()
do_sound = pygame.mixer.Sound("aikensa/core/config/sound/do.wav") 
re_sound = pygame.mixer.Sound("aikensa/core/config/sound/re.wav")
mi_sound = pygame.mixer.Sound("aikensa/core/config/sound/mi.wav")
fa_sound = pygame.mixer.Sound("aikensa/core/config/sound/fa.wav")
so_sound = pygame.mixer.Sound("aikensa/core/config/sound/sol.wav")
la_sound = pygame.mixer.Sound("aikensa/core/config/sound/la.wav")
si_sound = pygame.mixer.Sound("aikensa/core/config/sound/si.wav")
alarm_sound = pygame.mixer.Sound("aikensa/core/config/sound/mixkit-classic-short-alarm-993.wav")
picking_sound = pygame.mixer.Sound("aikensa/core/config/sound/mixkit-kids-cartoon-close-bells-2256.wav")
picking_sound_v2 = pygame.mixer.Sound("aikensa/core/config/sound/mixkit-page-forward-single-chime-1107.wav")
keisoku_sound = pygame.mixer.Sound("aikensa/core/config/sound/mixkit-bell-notification-933.wav") 
konpou_sound = pygame.mixer.Sound("aikensa/core/config/sound/mixkit-software-interface-back-2575.wav")
silence_sound = pygame.mixer.Sound("aikensa/core/config/sound/silence.wav")
ok_count_10_sound = pygame.mixer.Sound("aikensa/core/config/sound/mixkit-software-interface-remove-2576.wav")
ok_count_250_sound = pygame.mixer.Sound("aikensa/core/config/sound/tabanete.wav")
ok_sound = pygame.mixer.Sound("aikensa/core/config/sound/positive_interface.wav")
ng_sound = pygame.mixer.Sound("aikensa/core/config/sound/mixkit-classic-short-alarm-993.wav")

def _play_sound(sound, pre_silence=False):
    if pre_silence:
        channel = pygame.mixer.find_channel(True)
        if channel is not None:
            channel.play(silence_sound)
            channel.queue(sound)
            return
    sound.play()

def play_ok_sound():
    _play_sound(ok_sound)

def play_ng_sound():
    _play_sound(ng_sound)

def play_do_sound():
    _play_sound(do_sound)

def play_re_sound():
    _play_sound(re_sound)

def play_mi_sound():
    _play_sound(mi_sound)

def play_fa_sound():
    _play_sound(fa_sound)

def play_sol_sound():
    _play_sound(so_sound)

def play_la_sound():
    _play_sound(la_sound)

def play_si_sound():
    _play_sound(si_sound)

def play_alarm_sound():
    _play_sound(alarm_sound) 

def play_picking_sound():
    _play_sound(picking_sound_v2)

def play_keisoku_sound():
    _play_sound(keisoku_sound)

def play_konpou_sound(pre_silence=False):
    _play_sound(konpou_sound, pre_silence=pre_silence)

def play_ok_count_10_sound(pre_silence=False):
    _play_sound(ok_count_10_sound, pre_silence=pre_silence)

def play_ok_count_250_sound(pre_silence=False):
    _play_sound(ok_count_250_sound, pre_silence=pre_silence)