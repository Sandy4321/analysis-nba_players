# ----------------------------------------------------
# Collect player statistics from http://www.databasebasketball.com/
# 
# Sercan Taha Ahi, (tahaahi at gmail dot com)
#  Nov 2011
# July 2012
# ----------------------------------------------------
import urllib, urllib2
import string
import sys
import codecs

def parse_comp(key, year, n_active_players, fptr):
    url_list = 'http://www.databasebasketball.com/players/playerlist.htm?lt=' + key
    try:
        page = urllib2.urlopen(url_list).read()
    except:
        print 'Error: ' + key
        return 0
        
    ctr = page.count('<td><a href="playerpage.htm?ilkid=')
    print 'Surnames starting with ' + key + ': \t' + str(ctr)
    
    ptr1 = 0
    str1 = '<td><a href="playerpage.htm?ilkid='
    str2 = '">'
    str3 = '</a>'
    str4 = '<td class=cen>'
    str5 = '</td>'
    
    strp1 = 'Regular Season Stats'
    strp2 = '<td align="center">'
    strp3 = '</td>'
    strp4 = '<td align="right">'
    for i in xrange(0,ctr):
        ptr1 = page.find(str1, ptr1)
        ptr2 = page.find(str2, ptr1)
        ptr3 = page.find(str3, ptr2)
        ptr4 = page.find(str4, ptr3)
        ptr5 = page.find(str5, ptr4)
        
        player_code = page[ptr1+len(str1):ptr2-1]
        
        player_name = page[ptr2+len(str2):ptr3];
        player_name = player_name.replace('<b>', '')
        player_name = player_name.replace('</b>', '')
        player_name = player_name.replace('&nbsp;', ' ')
        player_name = player_name.split(', ');
        player_name = player_name[1] + " " + player_name[0]
        
        player_years = page[ptr4+len(str4):ptr5]
        player_year1 = int(player_years[0:4])
        player_year2 = int(player_years[7:12])
        
        if player_year1<= year and year+1 <= player_year2:
            ptr6 = page.find(str4, ptr5)
            ptr7 = page.find(str5, ptr6)
            player_position = page[ptr6+len(str4):ptr7]
            
            url_player = 'http://www.databasebasketball.com/players/playerpage.htm?ilkid=' + player_code
            try:
                page_player = urllib2.urlopen(url_player).read()
            except:
                print 'Error: Cannot access player page, ' + player_name
                break
            
            tmp = str(year+1)
            target_season = str(year) + '-' + tmp[2:4]
            ptrp0 = 0
            
            # Move to regular season stats
            ptrp0 = page_player.find(strp1, ptrp0)
            
            # Move to target season
            ptrp1 = page_player.find(target_season, ptrp0)
            ptrpx = page_player.find("All NBA First Team", ptrp0)
            if 0<ptrp1 and ptrp1<ptrpx:
                n_active_players = n_active_players + 1
                
                # Age
                ptrp1 = page_player.find(strp2, ptrp1)
                ptrp2 = page_player.find(strp3, ptrp1)
                player_age = int(page_player[ptrp1+len(strp2):ptrp2])
                
                # Team name
                tstr = 'yr=' + str(year) + '">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</a>', ptrp1)
                player_team = page_player[ptrp1+len(tstr):ptrp2]
                
                # Total played games
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_n_games = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Total played minutes
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_min = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Minutes per game
                player_minpg = float(player_min) / float(player_n_games)
                
                # Points
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_pts = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Points per game
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_ptspg = float(page_player[ptrp1+len(tstr):ptrp2])
                
                # Field goal made
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_fgm = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Field goal attempted
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_fga = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Field goal percentage
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_fgp = float(page_player[ptrp1+len(tstr):ptrp2])
                
                # Free throw made
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_ftm = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Free throw attemted
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_fta = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Free throw percentage
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_ftp = float(page_player[ptrp1+len(tstr):ptrp2])
                
                # 3-point made
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_3pm = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # 3-point attempted
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_3pa = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # 3-point percentage
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_3pp = float(page_player[ptrp1+len(tstr):ptrp2])
                
                # Offensive rebounds
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_orb = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Offensive rebounds per game
                player_orbpg = float(player_orb) / float(player_n_games)
                
                # Defensive rebounds
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_drb = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Offensive rebounds per game
                player_drbpg = float(player_drb) / float(player_n_games)
                
                # Total Rebounds
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_trb = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Total rebounds per game
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_trbpg = float(page_player[ptrp1+len(tstr):ptrp2])
                
                # Number of assists
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_ast = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Assists per game
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_astpg = float(page_player[ptrp1+len(tstr):ptrp2])
                
                # Number of steals
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_stl = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Steals per game
                player_stlpg = float(player_stl) / float(player_n_games)
                
                # Number of blocks
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_blk = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Blocks per game
                player_blkpg = float(player_blk) / float(player_n_games)
                
                # Number of turnovers
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_to = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Turnovers per game
                player_topg = float(player_to) / float(player_n_games)
                
                # Personal fouls
                tstr = '<td align="right">'
                ptrp1 = page_player.find(tstr, ptrp2)
                ptrp2 = page_player.find('</td>', ptrp1)
                player_pf = int(page_player[ptrp1+len(tstr):ptrp2])
                
                # Personal fouls per game
                player_pfpg = float(player_pf) / float(player_n_games)
                
                # Print the data
                fptr.write( player_name + "," )
                fptr.write( player_code + "," )
                fptr.write( player_position + "," )
                fptr.write( player_team + "," )
                fptr.write( str(player_year1) + "," )
                fptr.write( str(player_year2) + "," )
                fptr.write( str(player_age) + "," )
                fptr.write( str(player_n_games) + "," )
                fptr.write( str(player_min) + "," )
                fptr.write( str(player_minpg) + "," )
                fptr.write( str(player_pts) + "," )
                fptr.write( str(player_ptspg) + "," )
                fptr.write( str(player_fgm) + "," )
                fptr.write( str(player_fga) + "," )
                fptr.write( str(player_fgp) + "," )
                fptr.write( str(player_ftm) + "," )
                fptr.write( str(player_fta) + "," )
                fptr.write( str(player_ftp) + "," )
                fptr.write( str(player_3pm) + "," )
                fptr.write( str(player_3pa) + "," )
                fptr.write( str(player_3pp) + "," )
                fptr.write( str(player_orb) + "," )
                fptr.write( str(player_orbpg) + "," )
                fptr.write( str(player_drb) + "," )
                fptr.write( str(player_drbpg) + "," )
                fptr.write( str(player_trb) + "," )
                fptr.write( str(player_trbpg) + "," )
                fptr.write( str(player_ast) + "," )
                fptr.write( str(player_astpg) + "," )
                fptr.write( str(player_stl) + "," )
                fptr.write( str(player_stlpg) + "," )
                fptr.write( str(player_blk) + "," )
                fptr.write( str(player_blkpg) + "," )
                fptr.write( str(player_to) + "," )
                fptr.write( str(player_topg) + "," )
                fptr.write( str(player_pf) + "," )
                fptr.write( str(player_pfpg) + "\n" )
                
                ostr = str(n_active_players).rjust(4) + '\t' + player_name
                print ostr
        
        ptr1 = ptr5
        
    return n_active_players

    
def write_categories(fptr):
    fptr.write('Player,')
    fptr.write('Code,')
    fptr.write('Position,')
    fptr.write('Team,')
    fptr.write('From,')
    fptr.write('Until,')
    fptr.write('Age,')
    fptr.write('G,')
    fptr.write('MIN,')
    fptr.write('MINPG,')
    fptr.write('PTS,')
    fptr.write('PTSPG,')
    fptr.write('FGM,')
    fptr.write('FGA,')
    fptr.write('FGP,')
    fptr.write('FTM,')
    fptr.write('FTA,')
    fptr.write('FTP,')
    fptr.write('TPM,')
    fptr.write('TPA,')
    fptr.write('TPP,')
    fptr.write('ORB,')
    fptr.write('ORBPG,')
    fptr.write('DRB,')
    fptr.write('DRBPG,')
    fptr.write('TRB,')
    fptr.write('TRBPG,')
    fptr.write('AST,')
    fptr.write('ASTPG,')
    fptr.write('STL,')
    fptr.write('STLPG,')
    fptr.write('BLK,')
    fptr.write('BLKPG,')
    fptr.write('TO,')
    fptr.write('TOPG,')
    fptr.write('PF,')
    fptr.write('PFPG')
    fptr.write("\n")
    return
    

def main():
    if len(sys.argv) == 1:
        target_year = 1997
        #target_year = -1
    else:
        target_year = int(sys.argv[1])
    
    print "Collecting statistics for the " + str(target_year) + '-' + str(target_year+1) + " season...\n"
    fname = str(target_year) + '-' + str(target_year+1) + '.csv'
    fptr = codecs.open(fname, "w", "utf-8")
    write_categories(fptr)
    
    n_players = 0
    for key in range(ord('a'), ord('z')+1):
        n_players = parse_comp(chr(key), target_year, n_players, fptr)
    
    print '# of players = ' + str(n_players)
    fptr.close()

    
main()
