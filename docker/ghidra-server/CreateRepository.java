// Headless helper that seeds a default shared repository on the Ghidra Server.
//
// svrAdmin cannot create repositories (it only manages users and their access
// to repositories that already exist). Repositories are only created through
// the server's RMI client API, which is what a GUI Ghidra client does under the
// hood. This standalone program performs that same call headlessly so the
// container can provision a default repository on first boot.
//
// It initializes the Ghidra application, authenticates as the service account
// using a non-interactive password authenticator, and creates the repository
// (granting the service account ADMIN). It is idempotent: if the repository
// already exists it exits 0 without changes.
//
// Exit codes: 0 = repo present (created or pre-existing) or nothing to do;
//             1 = could not connect / authenticate (caller may retry).
import ghidra.GhidraApplicationLayout;
import ghidra.framework.Application;
import ghidra.framework.HeadlessGhidraApplicationConfiguration;
import ghidra.framework.client.ClientUtil;
import ghidra.framework.client.PasswordClientAuthenticator;
import ghidra.framework.client.RepositoryAdapter;
import ghidra.framework.client.RepositoryServerAdapter;
import ghidra.framework.remote.GhidraServerHandle;
import ghidra.framework.remote.User;

import java.util.Arrays;

public class CreateRepository {

    public static void main(String[] args) {
        String host = env("GHIDRA_SERVER_HOST", "localhost");
        int port = Integer.parseInt(env("GHIDRA_SERVER_PORT",
                Integer.toString(GhidraServerHandle.DEFAULT_PORT)));
        String user = env("GHIDRA_SERVER_USER", "agent");
        String password = System.getenv("GHIDRA_SERVER_PASSWORD");
        String repoName = env("GHIDRA_DEFAULT_REPOSITORY", "");

        if (repoName.isEmpty()) {
            System.out.println("[create-repo] GHIDRA_DEFAULT_REPOSITORY unset; nothing to do.");
            return;
        }
        if (password == null || password.isEmpty()) {
            System.err.println("[create-repo] GHIDRA_SERVER_PASSWORD unset; cannot authenticate. Skipping.");
            return;
        }

        try {
            // Bring up the Ghidra framework so the RMI/SSL client stack is usable.
            // GhidraApplicationLayout derives the install root from the location of
            // the Ghidra jars on the classpath.
            Application.initializeApplication(new GhidraApplicationLayout(),
                    new HeadlessGhidraApplicationConfiguration());

            // Supply credentials non-interactively (no GUI dialog, no stdin prompt).
            ClientUtil.setClientAuthenticator(new PasswordClientAuthenticator(user, password));

            System.out.printf("[create-repo] connecting to %s:%d as '%s'...%n", host, port, user);
            RepositoryServerAdapter server = ClientUtil.getRepositoryServer(host, port);
            if (!server.isConnected()) {
                server.connect();
            }
            if (!server.isConnected()) {
                System.err.println("[create-repo] could not connect to Ghidra Server (will retry).");
                System.exit(1);
            }

            if (Arrays.asList(server.getRepositoryNames()).contains(repoName)) {
                System.out.printf("[create-repo] repository '%s' already exists; nothing to do.%n", repoName);
                return;
            }

            System.out.printf("[create-repo] creating repository '%s'...%n", repoName);
            RepositoryAdapter repo = server.createRepository(repoName);
            // Grant the service account ADMIN; no anonymous access.
            repo.setUserList(new User[] { new User(user, User.ADMIN) }, false);
            System.out.printf("[create-repo] repository '%s' created; '%s' granted ADMIN.%n", repoName, user);
        }
        catch (Exception e) {
            System.err.println("[create-repo] failed: " + e);
            System.exit(1);
        }
    }

    private static String env(String key, String def) {
        String v = System.getenv(key);
        return (v == null || v.isEmpty()) ? def : v;
    }
}
