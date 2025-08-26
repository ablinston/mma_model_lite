# This function stops the script if the passed argument isn't True

def stop_if_not(check, check_name = "Check "):

    if check:
        # print(check_name + " PASSED")
        return
    else:
        print(check_name + " FAILED")
        raise SystemExit
