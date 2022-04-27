# drone_rl

Trochę informacji, pliki zazwyczaj robią to co mają w nazwie.  

W plikach continue_training.py, train_different_envs.py, test_network.py nalezy podać katalog z polityką oraz zapisaną iteracje (w funkcji main).  

Ogólnie kod się dynamicznie zmienia i część configów może nie działać z aktualnym kodem. Jeśli będzie wyrzucało błąd, że nie ma jakiś danych w configach to nalezy je dopisać, głównie będzie tu chodzić o sektor rewards oraz names w nim.

names muszą być po kolei tak jak w pliku vision_env.cpp (compureRewards()), natomiast wszystkie parametry są wczytywane w funkcji loadParam() w tym samym pliku.

Config przystosowany do szkolenia na obrazach: /configs/current.yaml.
  
  
Nagrody naliczają sie w każdym kroku symulacji.
  
Opis parametrów:

  x_vel_coeff: 0.006    -- nagroda = x_vel * x_vel_coeff
  collision_coeff: -1.8   -- jesli jest kolizja z kulką to nagroda = collision_coeff * e ^ -odleglosc od srodka
  touch_collision_coeff: -1.2   -- jesli jest kolizja z kulką to nagroda = touch_collision_coeff
  angular_vel_coeff: -5e-04     -- nagroda = angular_vel_coeff * norm(predkosci_katowe)
  survive_rew: 30.0   -- if x > survive_reward_distance: nagroda = survive_rew
  step_coeff: 0.0     --  nagroda = aktualny krok symulacji * step_coeff(raczej ujemny ale to chyba nie dzialalo za dobrze)
  distance_coef: 0.09     -- nagroda = x * distance_coef
  finish_rew: 100.0       -- if x > end_distance: nagroda = finish_rew -> reset
  too_low_coeff: 0.0      -- if z < 2: nagroda = too_low_coeff * abs(z - 2) (chyba)
  crash_penalty: -90.0    -- jesli wyleci poza world_box nagroda = crash_penalty-> reset
  timeout_penalty: -90.0    -- jesli timeout nagroda = timeout_penalty-> reset
  ball_crash_penalty: -90.0   -- jesli wleci w kulke i colision_is_crash nagroda = ball_crash_penalty-> reset
  colision_is_crash: yes  -- jesli yes to wlecenie wkulke resetuje env
  randomise_ball_position: no   -- zmienia pozycje o kulek +- 2, nie dziala z obrazem
  change_env_on_reset: no     -- zmienia env po kazdym resecie (pozycje kulek)  nie dziala z obrazem
  survive_reward_distance: 71.0   -- jesli x wiekszy od tego to dodaje sie survive_rew
  end_distance: 74.0    
  too_big_angle_penalty: -10    -- if abs(angle > 0.4): nagroda = too_big_angle_penalty
  backward_velocity_coeff: 3    -- if x_vel < 0: nagroda = x_vel * x_vel_coeff * backward_velocity_coeff

  names:
  - x_vel_reward
  - collision_penalty
  - touch_collision_penalty
  - ang_vel_penalty
  - survive_rew
  - step_penalty
  - distance_reward
  - too_low_penalty
  - too_big_angle_penalty
  - total
