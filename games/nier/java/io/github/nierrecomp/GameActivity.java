package io.github.nierrecomp;

import android.app.AlertDialog;
import android.app.NativeActivity;
import android.content.DialogInterface;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.drawable.GradientDrawable;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

/**
 * The game's activity: the runtime's NativeActivity (libnier.so, android_main) plus a small Menu button over the
 * picture. Tapping it asks "Exit game?"; yes finishes the activity, which ends the process (android_main) and brings
 * back the app's home screen, so the game no longer has to be swiped away from the recent apps.
 *
 * The button is a window of its own (a panel attached to the game's window): the game's touches come straight from the
 * window's input queue to native code, so a view added to the game's own window would never see a tap. Anything that
 * goes wrong here only means there is no button; the game is untouched. NX_EXIT_BUTTON=0 in config.txt: no button.
 */
public class GameActivity extends NativeActivity {
    private View menuButton;
    private AlertDialog dialog;
    private int attempts;

    @Override
    protected void onResume() {
        super.onResume();
        showMenuButton();
    }

    @Override
    protected void onPause() {
        removeMenuButton();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        removeMenuButton();
        super.onDestroy();
    }

    private static boolean buttonEnabled() {
        try (BufferedReader r = new BufferedReader(new FileReader(new File(Launcher.dataDir(), "config.txt")))) {
            String line;
            while ((line = r.readLine()) != null)
                if (line.trim().equals("NX_EXIT_BUTTON=0")) return false;
        } catch (Throwable ignored) {
        }
        return true;
    }

    private void showMenuButton() {
        if (menuButton != null || !buttonEnabled()) return;
        final View decor = getWindow().getDecorView();
        if (decor.getWindowToken() == null) {  // the window is not attached yet: look again shortly (10 s at most)
            if (attempts++ < 40)
                decor.postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        showMenuButton();
                    }
                }, 250);
            return;
        }
        try {
            float d = getResources().getDisplayMetrics().density;
            final float density = d;
            final TextView b = new TextView(this);
            b.setText(Lang.t("MENÚ", "MENU"));
            b.setTextColor(Color.WHITE);
            b.setTextSize(11);
            b.setGravity(Gravity.CENTER);
            int pad = (int) (8 * d);
            b.setPadding(pad, pad, pad, pad);
            GradientDrawable bg = new GradientDrawable();
            bg.setColor(Color.argb(120, 0, 0, 0));
            bg.setCornerRadius(10 * d);
            b.setBackground(bg);
            b.setAlpha(0.65f);
            final WindowManager wm = getWindowManager();
            final WindowManager.LayoutParams lp = new WindowManager.LayoutParams(
                    WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT,
                    WindowManager.LayoutParams.TYPE_APPLICATION_PANEL,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE | WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
                    PixelFormat.TRANSLUCENT);
            lp.token = decor.getWindowToken();
            lp.gravity = Gravity.TOP | Gravity.START;  // top-left corner by default
            lp.x = (int) (16 * d);
            lp.y = (int) (16 * d);
            lp.setTitle("NieR menu");
            b.setOnTouchListener(new View.OnTouchListener() {
                private int initialX, initialY;
                private float initialTouchX, initialTouchY;
                private boolean isMove;

                @Override
                public boolean onTouch(View v, android.view.MotionEvent event) {
                    switch (event.getAction()) {
                        case android.view.MotionEvent.ACTION_DOWN:
                            initialX = lp.x;
                            initialY = lp.y;
                            initialTouchX = event.getRawX();
                            initialTouchY = event.getRawY();
                            isMove = false;
                            return true;
                        case android.view.MotionEvent.ACTION_MOVE:
                            float dx = event.getRawX() - initialTouchX;
                            float dy = event.getRawY() - initialTouchY;
                            if (Math.hypot(dx, dy) > 8 * density) {
                                isMove = true;
                                lp.x = initialX + (int) dx;
                                lp.y = initialY + (int) dy;
                                try {
                                    wm.updateViewLayout(b, lp);
                                } catch (Throwable ignored) {
                                }
                            }
                            return true;
                        case android.view.MotionEvent.ACTION_UP:
                            if (!isMove) {
                                showExitDialog();
                            }
                            return true;
                    }
                    return false;
                }
            });
            wm.addView(b, lp);
            menuButton = b;
        } catch (Throwable t) {
            menuButton = null;
        }
    }

    private void removeMenuButton() {
        attempts = 0;
        if (dialog != null) {
            try {
                dialog.dismiss();
            } catch (Throwable ignored) {
            }
            dialog = null;
        }
        if (menuButton == null) return;
        try {
            getWindowManager().removeViewImmediate(menuButton);
        } catch (Throwable ignored) {
        }
        menuButton = null;
    }

    private void showExitDialog() {
        if (dialog != null && dialog.isShowing()) return;
        try {
            AlertDialog.Builder b = new AlertDialog.Builder(this, android.R.style.Theme_Material_Dialog_Alert);
            b.setTitle(Lang.t("Menú", "Menu"));
            b.setMessage(Lang.t("¿Salir del juego y volver al menú de la app? Se pierde lo que no hayas guardado.",
                    "Leave the game and go back to the app menu? Anything you have not saved is lost."));
            b.setPositiveButton(Lang.t("Salir del juego", "Exit game"), new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface d, int which) {
                    finish();
                }
            });
            b.setNegativeButton(Lang.t("Cancelar", "Cancel"), null);
            dialog = b.create();
            dialog.show();
        } catch (Throwable ignored) {
        }
    }
}
