# TODOs
 - add an option to organize the folder that binaries are uploaded to in ghidra through the website.
 - The upload popup has the cancel button in the bottom right. lets move that to upper right so it looks less like a submit button. take the opportunity to move any other buttons around that make sense in that popup.
 - when uploads fail, the error message is a bland grey. iots hard to notice if you leave and come back. it should be like red and bigger or something.give me some ideas.


# In Progress

# Completed
 - the select binary popup says "No open programs in Ghidra." even after you upload a binary. how do we actually start a session then? — fixed: engine now mounts the shared server-bound repo as a project at startup (fork commit 2ae45da, GHIDRA_SERVER_REPOSITORY); web picker lists via list_project_files and lazily opens via open_program. Verified e2e.