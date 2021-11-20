Data Path:./dataset/kr/FB15K/raw_data
Output Path:./dataset/kr/FB15K/experimental_output/TuckER2021-11-20--22-11-25.97
FB15K.zip file already exists in /home/hongbang/CogKTR/dataset/kr/FB15K/raw_data/FB15K.zip
+-------+------------+--------------------------------------------------------------------------+-----------+
| index |    head    |                                 relation                                 |    tail   |
+-------+------------+--------------------------------------------------------------------------+-----------+
|   0   |  /m/027rn  |                   /location/country/form_of_government                   |  /m/06cx9 |
|   1   | /m/017dcd  |       /tv/tv_program/regular_cast./tv/regular_tv_appearance/actor        | /m/06v8s0 |
|   2   | /m/07s9rl0 |                    /media_common/netflix_genre/titles                    | /m/0170z3 |
|   3   | /m/01sl1q  |      /award/award_winner/awards_won./award/award_honor/award_winner      | /m/044mz_ |
|   4   | /m/0cnk2q  | /soccer/football_team/current_roster./sports/sports_team_roster/position | /m/02nzb8 |
+-------+------------+--------------------------------------------------------------------------+-----------+
+-------+-----------+----------------------------------------------------------------------------------------------------------------------+------------+
| index |    head   |                                                       relation                                                       |    tail    |
+-------+-----------+----------------------------------------------------------------------------------------------------------------------+------------+
|   0   | /m/07pd_j |                                                   /film/film/genre                                                   | /m/02l7c8  |
|   1   |  /m/06wxw |                                            /location/location/time_zones                                             | /m/02fqwt  |
|   2   | /m/0d4fqn |                            /award/award_winner/awards_won./award/award_honor/award_winner                            | /m/03wh8kl |
|   3   | /m/07kcvl | /american_football/football_team/historical_roster./american_football/football_historical_roster_position/position_s | /m/0bgv8y  |
|   4   | /m/012201 |                                             /film/music_contributor/film                                             | /m/0ckrnn  |
+-------+-----------+----------------------------------------------------------------------------------------------------------------------+------------+
+-------+-----------+-------------------------------------------------------------------------------------+------------+
| index |    head   |                                       relation                                      |    tail    |
+-------+-----------+-------------------------------------------------------------------------------------+------------+
|   0   | /m/01qscs |         /award/award_nominee/award_nominations./award/award_nomination/award        | /m/02x8n1n |
|   1   |  /m/040db |                       /base/activism/activist/area_of_activism                      |  /m/0148d  |
|   2   |  /m/08966 | /travel/travel_destination/climate./travel/travel_destination_monthly_climate/month |  /m/05lf_  |
|   3   | /m/01hww_ |      /music/performance_role/regular_performances./music/group_membership/group     | /m/01q99h  |
|   4   |  /m/0c1pj |         /award/award_nominee/award_nominations./award/award_nomination/award        | /m/019f4v  |
+-------+-----------+-------------------------------------------------------------------------------------+------------+
+-------+------------+
| index |    name    |
+-------+------------+
|   0   |  /m/027rn  |
|   1   |  /m/06cx9  |
|   2   | /m/017dcd  |
|   3   | /m/06v8s0  |
|   4   | /m/07s9rl0 |
+-------+------------+
+-------+--------------------------------------------------------------------------+
| index |                                   name                                   |
+-------+--------------------------------------------------------------------------+
|   0   |                   /location/country/form_of_government                   |
|   1   |       /tv/tv_program/regular_cast./tv/regular_tv_appearance/actor        |
|   2   |                    /media_common/netflix_genre/titles                    |
|   3   |      /award/award_winner/awards_won./award/award_honor/award_winner      |
|   4   | /soccer/football_team/current_roster./sports/sports_team_roster/position |
+-------+--------------------------------------------------------------------------+
data_length:
 483142 50000 59071
table_length:
 14951 1345
Available cuda devices: 1
Epoch1/100   Train Loss: 0.325323113435941  Valid Loss: 0.1644868922164983
Epoch2/100   Train Loss: 0.14909171941856675  Valid Loss: 0.14284344194718943
Epoch3/100   Train Loss: 0.13593856021219136  Valid Loss: 0.13503723942181645
Epoch4/100   Train Loss: 0.12709783750358006  Valid Loss: 0.12606162138645302
Epoch5/100   Train Loss: 0.12023187107548414  Valid Loss: 0.12140275424589282
Evaluating Model TuckER...
